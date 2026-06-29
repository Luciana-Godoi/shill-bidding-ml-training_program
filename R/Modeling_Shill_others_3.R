# Foram comparados onze métodos de classificação: LDA, QDA, Árvore de Decisão, Bagging, Random Forest, 
# KNN, Regressão Logística, LASSO, SVM Linear, SVM Polinomial e SVM com núcleo RBF. Mesmo conjunto de 
# treinamento/teste para todos. Cada modelo foi submetido a validação cruzada estratificada 10-fold com 
# 3 repetições. Nos modelos que exigiam ajuste de hiperparâmetros (Árvore de Decisão, Random Forest, KNN 
# e LASSO), foi realizada busca em grade (grid search), sendo registrada a quantidade de combinações 
# avaliadas. As probabilidades preditas obtidas durante a validação cruzada foram utilizadas para 
# determinar um limiar ótimo de classificação (threshold), selecionado pela maximização do coeficiente 
# Kappa. Em seguida, cada modelo foi ajustado utilizando todo o conjunto de treinamento e aplicado ao 
# conjunto de teste independente. Para cada método foram registrados o tempo de validação cruzada, 
# o tempo de ajuste final e o tempo total de processamento. Além disso, foi calculada a área sob a curva 
# ROC durante a validação cruzada (ROC_AUC_CV).

################################################################################

## Libraries

################################################################################

library(tidymodels)
library(discrim)
library(baguette)
library(kernlab)
library(glmnet)
library(ranger)
library(gt)
library(dplyr)
library(readr)

################################################################################

## Data

################################################################################

set.seed(2635)

d_tr <- read_csv("train_set_complete.csv")
d_ts <- read_csv("test_set_complete.csv")

d_tr$Class <- factor(d_tr$Class, levels = c("1","0"))
d_ts$Class <- factor(d_ts$Class, levels = c("1","0"))

vars_remove <- c(
  "Intercept",
  "Last_Bidding",
  "Auction_Bids",
  "Starting_Price_Average",
  "Early_Bidding"
)

d_tr <- d_tr |> select(-all_of(vars_remove))
d_ts <- d_ts |> select(-all_of(vars_remove))


################################################################################

## Cross Validation

################################################################################

set.seed(2644)

folds <- vfold_cv(
  d_tr,
  v = 10,
  repeats = 3,
  strata = Class
)

################################################################################

## Best threshold

################################################################################

find_best_threshold <- function(
    truth,
    probs
){
  
  thresholds <- seq(
    0.001,
    0.999,
    by = 0.001
  )
  
  kappas <- sapply(
    thresholds,
    function(th){
      
      pred <- factor(
        ifelse(probs >= th,"1","0"),
        levels = c("1","0")
      )
      
      kap(
        tibble(
          truth = truth,
          pred = pred
        ),
        truth = truth,
        estimate = pred
      )$.estimate
      
    }
  )
  
  thresholds[which.max(kappas)]
  
}

################################################################################

## Metric Extraction Function

################################################################################

extrai_metricas <- function(
    preds,
    metodo,
    threshold
){
  
  cm_tbl <- conf_mat(
    preds,
    truth = Class,
    estimate = .pred_class
  )$table
  
  TP <- cm_tbl[1,1]
  FP <- cm_tbl[1,2]
  FN <- cm_tbl[2,1]
  TN <- cm_tbl[2,2]
  
  ACC <- (TP+TN)/(TP+TN+FP+FN)
  
  TPR <- TP/(TP+FN)
  
  TNR <- TN/(TN+FP)
  
  CSI <- TP/(TP+FP+FN)
  
  SSI <- TP/(TP+2*FP+2*FN)
  
  FAITH <- (TP+0.5*TN)/(TP+FP+FN+TN)
  
  PDIF <- (4*FP*FN)/(TP+FP+FN+TN)^2
  
  GS <- (TP*TN-FP*FN) /
    (
      (FN+FP)*(TP+FP+FN+TN)+
        (TP*TN-FP*FN)
    )
  
  GM <- sqrt(TPR*TNR)
  
  tibble(
    
    Metodo = metodo,
    
    Threshold = threshold,
    
    Accuracy = ACC,
    
    Sens = TPR,
    
    Spec = TNR,
    
    CSI = CSI,
    
    SSI = SSI,
    
    Faith = FAITH,
    
    PDIF = PDIF,
    
    GS = GS,
    
    MCC = mcc(
      preds,
      truth = Class,
      estimate = .pred_class
    )$.estimate,
    
    GMean = GM,
    
    Kappa = kap(
      preds,
      truth = Class,
      estimate = .pred_class
    )$.estimate,
    
    PPV = ppv(
      preds,
      truth = Class,
      estimate = .pred_class
    )$.estimate,
    
    NPV = npv(
      preds,
      truth = Class,
      estimate = .pred_class
    )$.estimate,
    
    J_Index = j_index(
      preds,
      truth = Class,
      estimate = .pred_class
    )$.estimate,
    
    Bal_Accuracy = bal_accuracy(
      preds,
      truth = Class,
      estimate = .pred_class
    )$.estimate,
    
    Detection_Prevalence =
      detection_prevalence(
        preds,
        truth = Class,
        estimate = .pred_class
      )$.estimate,
    
    Precision = precision(
      preds,
      truth = Class,
      estimate = .pred_class
    )$.estimate,
    
    Recall = recall(
      preds,
      truth = Class,
      estimate = .pred_class
    )$.estimate,
    
    F1 = f_meas(
      preds,
      truth = Class,
      estimate = .pred_class
    )$.estimate,
    
    ROC_AUC = roc_auc(
      preds,
      truth = Class,
      .pred_1
    )$.estimate,
    
    PR_AUC = pr_auc(
      preds,
      truth = Class,
      .pred_1
    )$.estimate
  )
  
}

################################################################################

## Fit Models

################################################################################

ajusta_modelo <- function(
    wf,
    metodo,
    folds,
    d_tr,
    d_ts,
    n_grid = 0
){
  
  cat("\n")
  cat("================================\n")
  cat(metodo,"\n")
  cat("================================\n")
  
  inicio_total <- Sys.time()
  
  inicio_cv <- Sys.time()
  
  cv_fit <- fit_resamples(
    wf,
    resamples = folds,
    control = control_resamples(
      save_pred = TRUE
    )
  )
  
  fim_cv <- Sys.time()
  
  tempo_cv <- as.numeric(
    difftime(
      fim_cv,
      inicio_cv,
      units = "mins"
    )
  )
  
  cv_preds <- collect_predictions(
    cv_fit
  )
  
  roc_auc_cv <- roc_auc(
    cv_preds,
    truth = Class,
    .pred_1
  )$.estimate
  
  thr <- find_best_threshold(
    truth = cv_preds$Class,
    probs = cv_preds$.pred_1
  )
  
  inicio_fit <- Sys.time()
  
  fit_final <- fit(
    wf,
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
  
  prob_test <- predict(
    fit_final,
    d_ts,
    type = "prob"
  )
  
  preds <- bind_cols(
    d_ts,
    prob_test
  ) |>
    mutate(
      .pred_class =
        factor(
          ifelse(
            .pred_1 >= thr,
            "1",
            "0"
          ),
          levels = c("1","0")
        )
    )
  
  resumo <- extrai_metricas(
    preds,
    metodo,
    thr
  )
  
  resumo <- resumo |>
    mutate(
      ROC_AUC_CV = roc_auc_cv
    )
  
  fim_total <- Sys.time()
  
  tempo_total <- as.numeric(
    difftime(
      fim_total,
      inicio_total,
      units = "mins"
    )
  )
  
  resumo <- resumo |>
    mutate(
      
      N_Treino = nrow(d_tr),
      
      N_Teste = nrow(d_ts),
      
      N_Combinacoes_Grid = n_grid,
      
      Tempo_CV_Min = round(
        tempo_cv,
        3
      ),
      
      Tempo_Fit_Min = round(
        tempo_fit,
        3
      ),
      
      Tempo_Total_Min = round(
        tempo_total,
        3
      )
      
    )
  
  list(
    resumo = resumo,
    modelo = fit_final,
    preds = preds
  )
  
}

################################################################################

## Linear discriminant analysis (LDA)

################################################################################

lda_wf <-
  workflow() |>
  add_formula(Class ~ .) |>
  add_model(
    discrim_linear() |>
      set_engine("MASS")
  )

lda_res <- ajusta_modelo(
  lda_wf,
  "LDA",
  folds,
  d_tr,
  d_ts
)

################################################################################

## Quadratic discriminant analysis (QDA)

################################################################################

qda_wf <-
  workflow() |>
  add_formula(Class ~ .) |>
  add_model(
    discrim_quad() |>
      set_engine("sparsediscrim")
  )

qda_res <- ajusta_modelo(
  qda_wf,
  "QDA",
  folds,
  d_tr,
  d_ts
)

################################################################################

## Bagging

################################################################################

bag_wf <-
  workflow() |>
  add_formula(Class ~ .) |>
  add_model(
    bag_tree() |>
      set_engine("rpart") |>
      set_mode("classification")
  )

bag_res <- ajusta_modelo(
  bag_wf,
  "Bagging",
  folds,
  d_tr,
  d_ts
)

################################################################################

## Logistic Regression

################################################################################

log_wf <-
  workflow() |>
  add_formula(Class ~ .) |>
  add_model(
    logistic_reg() |>
      set_engine("glm")
  )

log_res <- ajusta_modelo(
  log_wf,
  "Logistic",
  folds,
  d_tr,
  d_ts
)

################################################################################

## Decision Tree

################################################################################

tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) |>
  set_engine("rpart") |>
  set_mode("classification")

tree_grid <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 4
)

tree_wf <- workflow() |>
  add_formula(Class ~ .) |>
  add_model(tree_spec)

tree_tun <- tune_grid(
  tree_wf,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc)
)

tree_best <- select_best(
  tree_tun,
  metric = "roc_auc"
)

tree_wf_final <- finalize_workflow(
  tree_wf,
  tree_best
)
tree_res <- ajusta_modelo(
  tree_wf_final,
  "Tree",
  folds,
  d_tr,
  d_ts,
  n_grid = nrow(tree_grid)
)

################################################################################

## Random Forest

################################################################################

rf_grid <- tibble(
  mtry = 1:5
)

rf_spec <- rand_forest(
  mtry = tune(),
  trees = 1000
) |>
  set_engine("ranger",
             importance = "permutation") |>
  set_mode("classification")

rf_wf <- workflow() |>
  add_formula(Class ~ .) |>
  add_model(rf_spec)

rf_tun <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc)
)

rf_best <- select_best(
  rf_tun,
  metric = "roc_auc"
)

rf_wf_final <- finalize_workflow(
  rf_wf,
  rf_best
)

rf_res <- ajusta_modelo(
  rf_wf_final,
  "Random Forest",
  folds,
  d_tr,
  d_ts,
  n_grid = nrow(rf_grid)
)

################################################################################

## KNN

################################################################################

knn_spec <- nearest_neighbor(
  neighbors = tune()
) |>
  set_engine("kknn") |>
  set_mode("classification")

knn_grid <- tibble(
  neighbors = seq(1,51,2)
)

knn_wf <- workflow() |>
  add_formula(Class ~ .) |>
  add_model(knn_spec)

knn_tun <- tune_grid(
  knn_wf,
  resamples = folds,
  grid = knn_grid,
  metrics = metric_set(roc_auc)
)

knn_best <- select_best(
  knn_tun,
  metric = "roc_auc"
)

knn_wf_final <- finalize_workflow(
  knn_wf,
  knn_best
)

knn_res <- ajusta_modelo(
  knn_wf_final,
  "KNN",
  folds,
  d_tr,
  d_ts,
  n_grid = nrow(knn_grid)
)

################################################################################

## LASSO

################################################################################

lasso_spec <- logistic_reg(
  penalty = tune()
) |>
  set_engine("glmnet")

lasso_grid <- tibble(
  penalty = 10^seq(-5,1,length.out=30)
)

lasso_wf <- workflow() |>
  add_formula(Class ~ .) |>
  add_model(lasso_spec)

lasso_tun <- tune_grid(
  lasso_wf,
  resamples = folds,
  grid = lasso_grid,
  metrics = metric_set(roc_auc)
)

lasso_best <- select_best(
  lasso_tun,
  metric = "roc_auc"
)

lasso_wf_final <- finalize_workflow(
  lasso_wf,
  lasso_best
)

lasso_res <- ajusta_modelo(
  lasso_wf_final,
  "LASSO",
  folds,
  d_tr,
  d_ts,
  n_grid = nrow(
    collect_metrics(lasso_tun)
  )
)

################################################################################

## SVM

################################################################################

################################################################################
## SVM RBF
################################################################################
svm_rbf_wf <-
  workflow() |>
  add_formula(Class ~ .) |>
  add_model(
    svm_rbf(
      cost = 5,
      rbf_sigma = 0.25
    ) |>
      set_engine("kernlab") |>
      set_mode("classification")
  )

svm_rbf_res <- ajusta_modelo(
  svm_rbf_wf,
  "SVM RBF",
  folds,
  d_tr,
  d_ts
)

################################################################################
## SVM Linear
################################################################################

svm_linear_wf <-
  workflow() |>
  add_formula(Class ~ .) |>
  add_model(
    svm_linear() |>
      set_engine("kernlab") |>
      set_mode("classification")
  )

svm_linear_res <- ajusta_modelo(
  svm_linear_wf,
  "SVM Linear",
  folds,
  d_tr,
  d_ts
)

################################################################################
## SVM Polynomial
################################################################################

svm_poly_wf <-
  workflow() |>
  add_formula(Class ~ .) |>
  add_model(
    svm_poly(
      cost = 0.5,
      degree = 1
    ) |>
      set_engine("kernlab") |>
      set_mode("classification")
  )

svm_poly_res <- ajusta_modelo(
  svm_poly_wf,
  "SVM Polynomial",
  folds,
  d_tr,
  d_ts
)



################################################################################

## Final Comparison Table

################################################################################

resultados <- bind_rows(
  
  lda_res$resumo,
  
  qda_res$resumo,
  
  tree_res$resumo,
  
  bag_res$resumo,
  
  rf_res$resumo,
  
  knn_res$resumo,
  
  log_res$resumo,
  
  lasso_res$resumo,
  
  svm_rbf_res$resumo,
  
  svm_linear_res$resumo,
  
  svm_poly_res$resumo
  
)

resultados <- resultados |>
  select(
    
    Metodo,
    
    Threshold,
    
    N_Treino,
    
    N_Teste,
    
    N_Combinacoes_Grid,
    
    Tempo_CV_Min,
    
    Tempo_Fit_Min,
    
    Tempo_Total_Min,
    
    ROC_AUC_CV,
    
    everything()
    
  ) |>
  arrange(
    desc(MCC),
    desc(PR_AUC),
    desc(ROC_AUC)
  )

write.csv(
  resultados,
  "comparacao_modelos_varios.csv",
  row.names = FALSE
)

resultados

resultados_transp <- resultados |>
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
## GT Table
################################################################################

resultados_transp |>
  gt() |>
  fmt_number(
    columns = where(is.numeric),
    decimals = 5
  ) |>
  tab_header(
    title = "Comparison of methods - CV"
  )

saveRDS(
  resultados_transp,
  "comparacao_metodos_varios_CV.rds"
)



