################################################################################
## Libraries
################################################################################

library(tidymodels)
library(themis)
library(doParallel)
library(gt)
library(dplyr)
library(tidyr)
library(janitor)
library(stringr)
library(vip)
library(probably)
library(gt)
library(lubridate)
library(readr)
library(summarytools)
library(xgboost)
library(ggplot2)
library(gtsummary)
library(rstatix)
library(shapviz)

#############################
# Loading data ##############
#############################

# Define-se que se deve buscar os dados na pasta em que está salvo p .R ou .Rmd 
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

Data_shillB <- readr::read_csv("Data_shillB.csv") # 6321 * 13
names(Data_shillB)

d_shillB <- Data_shillB |> 
  dplyr::select(
    -Record_ID, -Auction_ID, -Bidder_ID
  ) |> # 6321 por 10
  drop_na() # 6321  por  10

with(d_shillB, freq(Class), useNA="yes")

### Z-score

d_shillB_pad<- d_shillB |> 
  dplyr::mutate(
    # Padroniza todas as colunas numéricas, EXCETO a coluna 'Class'
    dplyr::across(where(is.numeric) & -Class, ~ as.vector(scale(.)))
  )

# Auxiliary functions -----------------------------------------------------

## Cohen's D: compute the effect size for t-test  ####
my_cohen_d <- function(data, variable, by, ...) {
  sprintf("%.2f", rstatix::cohens_d(data, as.formula(glue::glue("{variable} ~ {by}")))$effsize)
}

## Cramer's V: measures the strength of the association between categorical variables ####
my_cramer_v <- function(data, variable, by, ...) {
  sprintf("%.2f", table(data[[variable]], data[[by]]) |> rstatix::cramer_v())
}

#####################################
# Descriptive analysis ##############
#####################################

# Descriptive analysis ----------------------------------------------------

## Table 1 - Features of the Shill Bidding dataset ####
tbl01 <- d_shillB |>
  dplyr::mutate(
    Class = factor(
      Class,
      levels = c(0, 1),
      labels = c("Normal", "Suspicious")
    )
  ) |>
  tbl_summary(
    include = c(Bidder_Tendency, Bidding_Ratio, Successive_Outbidding, Last_Bidding, 
    Early_Bidding, Winning_Ratio, Auction_Duration
                ),
    label = list(
      Bidder_Tendency ~ "Bidder Tendency; mean ± SD",
      Bidding_Ratio ~ "Bidding Ratio; mean ± SD",
      Successive_Outbidding ~ "Successive Outbidding; N (%)",
      Last_Bidding ~ "Last Bidding; mean ± SD",
      Early_Bidding ~ "Early Bidding; mean ± SD",
      Winning_Ratio ~ "Winning Ratio; mean ± SD",
      Auction_Duration ~ "Auction Duration; N (%)"
    ),
    by = Class,
    statistic = list(
      all_continuous() ~ "{mean} ({sd})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = list(
      all_continuous() ~ 2, 
      all_categorical() ~ c(0, 1)
    )
  ) |> 
  add_p() |> 
  add_stat(
    fns = list(
      all_continuous() ~ my_cohen_d,
      all_categorical() ~ my_cramer_v
    )
  ) |> 
  modify_header(
    label ~ "**Features**",
    p.value ~ "**p**",
    add_stat_1 ~ "**Effect size**"
  ) |> 
  bold_labels() |>
  modify_footnote(update = everything() ~ NA) |> 
  as_gt() |> 
  gt::tab_source_note(
    gt::md(
      "SD: standard deviation; p: p-value. Continuous variables are presented as mean (SD) and categorical variables as n (%). Effect size was estimated using Cohen's d for continuous variables and Cramer's V for categorical variables."
    )
  )

tbl01

wilcox.test(Bidding_Ratio ~ Class, data = d_shillB)

wilcox.test(Winning_Ratio ~ Class, data = d_shillB)

wilcox.test(Bidder_Tendency ~ Class, data = d_shillB)

chisq.test(table(d_shillB$Successive_Outbidding,
                 d_shillB$Class))

chisq.test(table(d_shillB$Auction_Duration,
                 d_shillB$Class))

cohens_d(d_shillB_pad, Bidding_Ratio~Class)


################################################################################
## XGBoost Specification -  Complete dataset 
################################################################################

d_shillB_pad <- d_shillB_pad |>
  dplyr::mutate(
    Class = factor(
      Class,
      levels = c(0, 1),
 #     labels = c("Normal", "Suspicious")
    )
  )

with(d_shillB_pad, freq(Class), useNA="yes")

rec <- recipe(Class ~ ., data = d_shillB_pad)

## Cross Validation
set.seed(2675)

xgb_folds <- vfold_cv(
  d_shillB_pad,
  v = 10,
  repeats = 3, 
  strata = Class 
)

## XGBoost Specification
xgb_espec <- boost_tree(
  trees = 1000,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) |>
  set_engine(
    "xgboost",
    nthread = 1
  ) |>
  set_mode("classification")

# workflow
xgb_wf <- workflow() |>
  add_recipe(rec) |>
  add_model(xgb_espec)

# tuning
metricas <- metric_set(
  roc_auc,
  accuracy,
  kap,
  brier_class
)

## Grid Search
xgb_grid <- grid_max_entropy(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), d_shillB_pad),
  learn_rate(),
  size = 30
)

################################################################################
## Parallel Processing
################################################################################

registerDoParallel(cores = 16)

xgb_tun <- tune_grid(
  xgb_wf,
  resamples = xgb_folds,
  grid = xgb_grid,
  metrics = metricas,
  control = control_grid(save_pred = TRUE)
)

# best parameters
best_par <- select_best(xgb_tun, metric = "kap") # "roc_auc"

wf <- workflow() |>
  add_recipe(rec) |>
  add_model(xgb_espec)

# workflow final
wf_final <- finalize_workflow(
  wf,
  best_par
)

# fit on complete dataset
xgb_fit <- fit(
  wf_final,
  data = d_shillB_pad
)

# Shap graphs

X <- d_shillB_pad |>
  dplyr::select(-Class)

modelo_xgb <- extract_fit_parsnip(xgb_fit)$fit

class(modelo_xgb)

sv <- shapviz(
  modelo_xgb,
  X_pred = data.matrix(X),
  X = X
)

sv_importance(sv)
sv_importance(sv, kind = "beeswarm")

#################################################################################################################### 
# Interpretação: A comparação entre o modelo Bayesiano Power Cauchy e o modelo XGBoost via Shap mostrou concordância
# quanto às variáveis mais relevantes: Successive_Outbidding e Winning_Ratio. As variáveis Auction_Duration e 
# Bidder_Tendency também demonstraram relevância consistente nos dois métodos, embora com menor magnitude. 
# Bidding_Ratio apresentou elevada importância no XGBoost (importância 3), mas seu IC de 95% HPD e percentílico
# incluiu o valor zero no modelo Bayesiano, sugerindo que sua contribuição pode estar associada a relações 
# não lineares ou interações com outras variáveis, características capturadas pelo XGBoost, mas não pelo modelo Bayesiano
# com estrutura linear. Já as variáveis Auction_Bids, Last_Bidding, Early_Bidding e Starting_Price_Average apresentaram 
# pouca improtância em ambas as metodologias. 
#################################################################################################################### 


################################################################################
## XGBoost Specification -  Train/Test dataset 
################################################################################

################################################################################
## Libraries
################################################################################

library(tidymodels)
library(themis)
library(doParallel)
library(gt)
library(dplyr)
library(tidyr)
library(shapviz)
library(ggplot2)

################################################################################
## Train/Test Split
################################################################################

set.seed(2635)

d_tr <- readr::read_csv("train_set_complete.csv") 
d_tr$Class <- factor(d_tr$Class, levels = c("1","0"))

d_ts <- readr::read_csv("test_set_complete.csv") 
d_ts$Class <- factor(d_ts$Class, levels = c("1","0"))

d_tr <- d_tr |> 
  dplyr::select(
    -Intercept, -Last_Bidding, -Auction_Bids, -Starting_Price_Average, -Early_Bidding
  ) 

d_ts <- d_ts |> 
  dplyr::select(
    -Intercept, -Last_Bidding, -Auction_Bids, -Starting_Price_Average, -Early_Bidding
  ) 


################################################################################
## Cross Validation
################################################################################

set.seed(2645)

xgb_folds <- vfold_cv(
  d_tr,
  v = 10,
  repeats = 3,
  strata = Class
)

################################################################################
## XGBoost Specification
################################################################################

xgb_espec <- boost_tree(
  trees = 1000,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) |>
  set_engine(
    "xgboost",
    nthread = 1
  ) |>
  set_mode("classification")

################################################################################
## Grid Search
################################################################################

xgb_grid <- grid_max_entropy(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), d_tr),
  learn_rate(),
  size = 30
)

################################################################################
## Parallel Processing
################################################################################

registerDoParallel(cores = 16)

################################################################################
## Function
################################################################################

ajusta_xgb <- function(recipe_obj, metodo){
  
  nome <- gsub(" ", "_", metodo)
  
  cat("\n")
  cat("=========================================================\n")
  cat("Método:", metodo, "\n")
  cat("=========================================================\n")
  
  inicio_total <- Sys.time()
  
  ###########################################################################
  ## Workflow
  ###########################################################################
  
  xgb_wf <- workflow() |>
    add_recipe(recipe_obj) |>
    add_model(xgb_espec)
  
  ###########################################################################
  ## Tuning
  ###########################################################################
  
  # tuning
  metricas <- metric_set(
    roc_auc,
    accuracy,
    kap,
    brier_class
  )
  
  inicio_tune <- Sys.time()
  
  set.seed(2689)
  
  xgb_tun <- tune_grid(
    xgb_wf,
    resamples = xgb_folds,
    grid = xgb_grid,
    metrics = metricas,
    control = control_grid(
    save_pred = TRUE
    )
  )
  
  fim_tune <- Sys.time()
  
  tempo_tune <- as.numeric(
    difftime(
      fim_tune,
      inicio_tune,
      units = "mins"
    )
  )
  
  ###########################################################################
  ## Best Hyperparameters
  ###########################################################################
  

  xgb_hips <- select_best(
    xgb_tun,
    metric = "kap"
  )
  
  pred_cv <- collect_predictions(
    xgb_tun,
    parameters = xgb_hips
  )
  
  names(pred_cv)
  
  saveRDS(
    xgb_hips,
    paste0(
      "hiperparametros_Optimal threshold using CV_",
      nome,
      ".rds"
    )
  )
  
  ###########################################################################
  ## Best Threshold Using Cross-Validation
  ###########################################################################
  
  find_best_threshold <- function(
    truth,
    probs
  ){
    
    thresholds <- seq(0.001, 0.999, by = 0.001)
    
    kap_values <- sapply(thresholds, function(th){
      
      pred <- factor(
        ifelse(probs >= th, "1", "0"),
        levels = c("1","0")
      )
      
      kap(
        tibble(
          truth = truth,
          estimate = pred
        ),
        truth = truth,
        estimate = estimate
      ) |> pull(.estimate)
      
    })
    
    thresholds[which.max(kap_values)]
    
  }
  
  thr_xgb <- find_best_threshold(
    truth = pred_cv$Class,
    probs = pred_cv$.pred_1
  )
  
  cat("\nThreshold ótimo (CV):\n")
  print(thr_xgb)
  
  
  ###########################################################################
  ## Final Model
  ###########################################################################
  
  xgb_wf_final <- finalize_workflow(
    xgb_wf,
    xgb_hips
  )
  
  inicio_fit <- Sys.time()
  
  xgb_final <- fit(
    xgb_wf_final,
    d_tr
  )
  
    fim_fit <- Sys.time()
  
  tempo_fit <- as.numeric(
    difftime(
      fim_fit,
      inicio_fit,
      units = "mins"
    )
  )
  
  fim_total <- Sys.time()
  
  tempo_total <- as.numeric(
    difftime(
      fim_total,
      inicio_total,
      units = "mins"
    )
  )
  
  
  
  ###############################################################################
  ## SHAP VALUES
  ###############################################################################
  
  cat("\nCalculando SHAP values...\n")
  
  # Extrai objeto xgboost
  xgb_fit <- extract_fit_parsnip(
    xgb_final
  )$fit
  
  # Prepara a base exatamente como o modelo enxergou
  rec_prep <- prep(
    recipe_obj,
    training = d_tr
  )
  
  X_shap <- bake(
    rec_prep,
    new_data = d_ts
  ) |>
    dplyr::select(-Class)
  
  # SHAP
  sv <- shapviz(
    xgb_fit,
    X_pred = as.matrix(X_shap),
    X = X_shap
  )
  
  saveRDS(
    sv,
    paste0("shap_Optimal threshold using CV_", nome, ".rds")
  )
  
  shap_bar <- sv_importance(sv)
  
  shap_beeswarm <- sv_importance(
    sv,
    kind = "beeswarm"
  )
  
  ggsave(
    filename = paste0(
      "SHAP_Beeswarm_Optimal threshold using CV_",
      nome,
      ".png"
    ),
    plot = shap_beeswarm,
    width = 10,
    height = 7,
    dpi = 300
  )
  
  ggsave(
    filename = paste0(
      "SHAP_Bar_Optimal threshold using CV_",
      nome,
      ".png"
    ),
    plot = shap_bar,
    width = 10,
    height = 7,
    dpi = 300
  )
  
  ###########################################################################
  ## Save Objects
  ###########################################################################
  
  saveRDS(
    xgb_final,
    paste0("modelo_Optimal threshold using CV_", nome, ".rds")
  )
  
  ###########################################################################
  ## Predictions
  ###########################################################################
  
  preds_prob <- predict(
    xgb_final,
    new_data = d_ts,
    type = "prob"
  )
  
  preds_class <- tibble(
    .pred_class = factor(
      ifelse(
        preds_prob$.pred_1 >= thr_xgb,
        "1",
        "0"
      ),
      levels = c("1","0")
    )
  )
  
  preds <- bind_cols(
    d_ts,
    preds_class,
    preds_prob
  )
  
  cat("\n====================\n")
  cat("COLUNAS DE PREDS\n")
  cat("====================\n")
  print(names(preds))
  
  cat("\n====================\n")
  cat("PRIMEIRAS LINHAS\n")
  cat("====================\n")
  print(head(preds))
  
  saveRDS(
    preds,
    paste0("preds_Optimal threshold using CV_", nome, ".rds")
  )
  
  
  # preds <- collect_predictions(
  #    xgb_final
  #  )
  
  #  saveRDS(
  #    preds,
  #    paste0("preds_", nome, ".rds")
  #  )
  
  ###########################################################################
  ## Metrics from tidymodels
  ###########################################################################
  
  mets <- bind_rows(
    
    accuracy(
      preds,
      truth = Class,
      estimate = .pred_class
    ),
    
    kap(
      preds,
      truth = Class,
      estimate = .pred_class
    )
    
  )
  
  saveRDS(
    mets,
    paste0("metrics_Optimal threshold using CV_", nome, ".rds")
  )
  
  ###########################################################################
  ## Confusion Matrix Metrics
  ###########################################################################
  
  cm <- summary(
    conf_mat(
      preds,
      truth = Class,
      estimate = .pred_class
    )
  )
  
  cm_wide <- cm |>
    select(.metric, .estimate) |>
    pivot_wider(
      names_from = .metric,
      values_from = .estimate
    )
  

  ###########################################################################
  ## ROC AUC
  ###########################################################################
  
  roc_auc_val <- roc_auc(
    preds,
    truth = Class,
    .pred_1
  )$.estimate
  
  ###########################################################################
  ## PR AUC
  ###########################################################################
  
  pr_auc_val <- pr_auc(
    preds,
    truth = Class,
    .pred_1
  )$.estimate
  
  ################################################################################
  ## Additional metrics 
  ################################################################################
  
  calc_metrics_imb <- function(preds){
    
    cm_tbl <- conf_mat(
      preds,
      truth = Class,
      estimate = .pred_class
    )$table
    
    TP <- cm_tbl[1,1]
    FP <- cm_tbl[1,2]
    FN <- cm_tbl[2,1]
    TN <- cm_tbl[2,2]
    
    ACC <- (TP + TN)/(TP + TN + FP + FN)
    
    TPR <- TP/(TP + FN)
    
    TNR <- TN/(TN + FP)
    
    CSI <- TP/(TP + FP + FN)
    
    SSI <- TP/(TP + 2*FP + 2*FN)
    
    FAITH <- (TP + 0.5*TN)/
      (TP + FP + FN + TN)
    
    PDIF <- (4*FP*FN)/
      (TP + FP + FN + TN)^2
    
    GS <- (TP*TN - FP*FN)/
      (
        (FN + FP)*
          (TP + FP + FN + TN) +
          (TP*TN - FP*FN)
      )
    
    GM <- sqrt(TPR*TNR)
    
    tibble(
      Accuracy = ACC,
      Sens = TPR,
      Spec = TNR,
      CSI = CSI,
      SSI = SSI,
      Faith = FAITH,
      PDIF = PDIF,
      GS = GS,
      GMean = GM
    )
  }
  
  imb_metrics <- calc_metrics_imb(preds)
  
  ###########################################################################
  ## Save Summary
  ###########################################################################
  
  resumo <- tibble(
    
    Metodo = metodo,
    
    Threshold = thr_xgb,
    
    N_Treino = nrow(d_tr),
    N_Teste = nrow(d_ts),
    
    N_Combinacoes_Grid = nrow(as.data.frame(xgb_grid)),
    
    Tempo_Tuning_Min = tempo_tune,
    Tempo_LastFit_Min = tempo_fit,
    Tempo_Total_Min = tempo_total,
    
    Accuracy = imb_metrics$Accuracy,
    
    Sens = imb_metrics$Sens,
    
    Spec = imb_metrics$Spec,
    
    CSI = imb_metrics$CSI,
    
    SSI = imb_metrics$SSI,
    
    Faith = imb_metrics$Faith,
    
    PDIF = imb_metrics$PDIF,
    
    GS = imb_metrics$GS,
    
    MCC = cm_wide$mcc,
    
    GMean = imb_metrics$GMean,
    
    Kappa = cm_wide$kap,
    
    PPV = cm_wide$ppv,
    
    NPV = cm_wide$npv,
    
    J_Index = cm_wide$j_index,
    
    Bal_Accuracy = cm_wide$bal_accuracy,
    
    Detection_Prevalence =
      cm_wide$detection_prevalence,
    
    Precision = cm_wide$precision,
    
    Recall = cm_wide$recall,
    
    F1 = cm_wide$f_meas,
    
    ROC_AUC = roc_auc_val,
    
    PR_AUC = pr_auc_val
  )
  
  saveRDS(
    resumo,
    paste0("resumo_Optimal threshold using CV_", nome, ".rds")
  )
  
  return(resumo)
  
}

################################################################################
## Recipes
################################################################################

## 1 - No balancing

rec_sem <- recipe(
  Class ~ .,
  data = d_tr
) |>
  step_dummy(
    all_nominal_predictors()
  )

## 2 - SMOTENC

rec_smotenc <- recipe(
  Class ~ .,
  data = d_tr
) |>
  step_smotenc(
    Class,
    over_ratio = 0.5
  ) |>
  step_dummy(
    all_nominal_predictors()
  )

## 3 - ADASYN

rec_adasyn <- recipe(
  Class ~ .,
  data = d_tr
) |>
  step_dummy(
    all_nominal_predictors()
  ) |>
  step_adasyn(
    Class,
    over_ratio = 0.5
  )

## 4 - BSMOTE

rec_bsmote <- recipe(
  Class ~ .,
  data = d_tr
) |>
  step_dummy(
    all_nominal_predictors()
  ) |>
  step_bsmote(
    Class,
    over_ratio = 0.5
  )

## 5 - ROSE

rec_rose <- recipe(
  Class ~ .,
  data = d_tr
) |>
  step_dummy(
    all_nominal_predictors()
  ) |>
  step_rose(
    Class
  )

################################################################################
## Run Models
################################################################################

res_sem     <- ajusta_xgb(rec_sem,     "Sem balanceamento")
res_smotenc <- ajusta_xgb(rec_smotenc, "SMOTENC")
res_adasyn  <- ajusta_xgb(rec_adasyn,  "ADASYN")
res_bsmote  <- ajusta_xgb(rec_bsmote,  "BSMOTE")
res_rose    <- ajusta_xgb(rec_rose,    "ROSE")

################################################################################
## Final Comparison Table
################################################################################

tbl_comp <- bind_rows(
  res_sem,
  res_smotenc,
  res_adasyn,
  res_bsmote,
  res_rose
)

tbl_comp_transp <- tbl_comp |>
  pivot_longer(
    cols = -Metodo,
    names_to = "Metrica",
    values_to = "Valor"
  ) |>
  pivot_wider(
    names_from = Metodo,
    values_from = Valor
  )

################################################################################
## Save comparison table
################################################################################

write.csv(
  tbl_comp,
  "comparacao_metodos_balanceamento_Optimal threshold using CV.csv",
  row.names = FALSE
)


saveRDS(
  tbl_comp,
  "comparacao_metodos_balanceamento_Optimal threshold using CV.rds"
)


################################################################################
## GT Table
################################################################################

tbl_comp_transp |>
  gt() |>
  fmt_number(
    columns = where(is.numeric),
    decimals = 4
  ) |>
  tab_header(
    title = "Comparison of balancing methods - XGBOOST (TRH - CV)"
  )

saveRDS(
  tbl_comp_transp,
  "comparacao_metodos_balanceamento_Optimal threshold using CV.rds"
)

################################################################################
## Load saved results
################################################################################


preds_adasyn_OC <- readRDS(
  "preds_Optimal threshold using CV_ADASYN.rds"
)

modelo_smotenc_OC <- readRDS(
  "modelo_Optimal threshold using CV_SMOTENC.rds"
)

hips_bsmote_OC <- readRDS(
  "hiperparametros_Optimal threshold using CV_BSMOTE.rds"
)

tbl_comp_OC <- readRDS(
  "comparacao_metodos_balanceamento_Optimal threshold using CV.rds"
)




























