##### Executar direto no console R

library(reticulate)

py_require(c(
  "numpy",
  "pandas",
  "scikit-learn",
  "pystan", 
  "arviz", 
  "matplotlib",
  "statsmodels"
))

###### Este programa foi reescrito a partir de uma implementação original em PyStan 2, 
###### tendo sido ajustado para a sintaxe e API do PyStan 3.10.1. Aqui a análise foi feita com todas
###### as covariáveis, sem a exclusão de exclusão das variáveis ['Class','Record_ID','
###### Auction_ID','Bidder_ID']. Também foi inserido código para extração das bases treino e teste.
###### Criação de gráficos de autocorrelação e trace plot dos parâmetros. 

# library
import pandas as pd
import numpy as np
import stan #pystan
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import arviz as az

## FUNCTIONS
# define min max scaler
scaler = MinMaxScaler()

# define standard scaler
scaler = StandardScaler()

## COMPLETE DATA SET
data = pd.read_csv("Data_shillB.csv", delimiter=",")
data.columns
X1 = data.drop(['Class','Record_ID','Auction_ID','Bidder_ID'], axis=1)
Xs = scaler.fit_transform(X1)
Xs = pd.DataFrame(Xs, columns=X1.columns)
Y = data['Class']

#dataset
N = Xs.shape[0]
x0  = np.ones(N)
X = np.c_[x0, np.array(Xs)]

# data for Stan
dataS = {}
dataS['y']   = Y
dataS['X']  = X
dataS['p']   = X.shape[1]
dataS['N']  = N

# Modelo Power Cauchy in stan
model_pc = '''
data {
    int<lower=0> p;
    int<lower=0> N;

    array[N] int<lower=0, upper=1> y;

    matrix[N,p] X;
}
parameters{
	vector[p] beta;
    real loglambda;
}
transformed parameters{
	vector[N] prob;
    vector[N] eta;
    real<lower = 0> lambda;
    lambda = exp(loglambda);
    eta = X*beta;
    for(i in 1:N){
    prob[i] = pow(cauchy_cdf(eta[i] | 0, 1), lambda);
    }
    }
model {
	beta ~ normal(0.0,100);
	loglambda ~ uniform(-2,2);
	y ~ bernoulli(prob);
}
generated quantities {
  vector[N] log_lik;
  for (i in 1:N) {
    // preferred Stan syntax as of version 2.10.0
    log_lik[i] = bernoulli_lpmf(y[i] | prob[i]);
  }
}
'''

# data for Stan
dataS = {}

dataS["y"] = np.asarray(Y).astype(int)
dataS["X"] = np.asarray(X).astype(float)

dataS["p"] = int(X.shape[1])
dataS["N"] = int(X.shape[0])

# mcmc values
chains = 2
iters = 10000
warmup = 5000
thin = 5 # 50
seed = 10000003

# =====================================
# POWER CAUCHY - PyStan 3.10.1
# =====================================

posterior_pc = stan.build(
    model_pc,
    data=dataS,
    random_seed=seed
)

fit_pc = posterior_pc.sample(
    num_chains=chains,
    num_samples=int((iters - warmup) / thin)
)

samples = fit_pc.to_frame()

print(samples.columns.tolist())

mpc_summary = samples.describe().T

print(mpc_summary.head())


### Credibility interval 95%

# 1. Filter the target columns
beta_cols = [c for c in samples.columns if c.startswith("beta")]

# 2. Calculate basic statistics and quantile intervals
ic95 = pd.DataFrame({
    "Parametro": beta_cols,
    "Media": samples[beta_cols].mean().values,
    "DP": samples[beta_cols].std().values,
    "IC95_inf": samples[beta_cols].quantile(0.025).values,
    "IC95_sup": samples[beta_cols].quantile(0.975).values
})

print(ic95)

#3. Create the Design Matrix (X)
x0 = np.ones(N)
X = np.c_[x0, np.array(Xs)]

#4. Generate descriptive parameter names
nomes = ["Intercept"] + list(Xs.columns)

#5. Overwrite and finalize the labels
ic95["Parametro"] = nomes

print(ic95)

### HPD interval 95%

# 1. Filter the target columns
beta_cols = [f"beta.{i+1}" for i in range(dataS["p"])]

# 2.Get the exact bounds of the raw HDI
hdi_bounds = [az.hdi(samples[col].values, prob=0.95) for col in beta_cols]
valores_inf = [bound[0] for bound in hdi_bounds]
valores_sup = [bound[1] for bound in hdi_bounds]

# 3. Map the parameter names
nomes = ["Intercept"] + list(Xs.columns)

# 5. Create the final table
hpd_table = pd.DataFrame({
    "Parametro": nomes,
    "Media": samples[beta_cols].mean().values,
    "DP": samples[beta_cols].std().values,
    "HPD_95_inf": valores_inf,
    "HPD_95_sup": valores_sup
})

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n=== TABELA HPD 95% CORRIGIDA (TODOS OS BETAS) ===")
print(hpd_table.round(4))

# ===============================================================================================
# Presents and computes both the percentile credible interval and the HPD interval simultaneously
# ===============================================================================================

# 1. Filter the target columns
beta_cols = [f"beta.{i+1}" for i in range(dataS["p"])]

# 2. Get the exact bounds of the raw HDI
hdi_bounds = [az.hdi(samples[col].values, prob=0.95) for col in beta_cols]
valores_inf = [bound[0] for bound in hdi_bounds]
valores_sup = [bound[1] for bound in hdi_bounds]

# 3. Map the parameter names
nomes = ["Intercept"] + list(Xs.columns)

# 4. Final table 
tabela_final = pd.DataFrame({
    "Parametro": nomes,
    "Media": samples[beta_cols].mean().values,
    "DP": samples[beta_cols].std().values,
    "IC95_inf": samples[beta_cols].quantile(0.025).values,
    "IC95_sup": samples[beta_cols].quantile(0.975).values,
    "HPD_95_inf": valores_inf,
    "HPD_95_sup": valores_sup
})

tabela_final = tabela_final.round(4)

print("\n=== Final Table (95%) ===")
print(tabela_final)

#### Resultado final dos intervalos percemtílicos e HDI equivalentes, saem as variáveis: Bidding_Ratio, Last_Bidding, 
# Auction_Bids, Starting_Price_Average, Early_Bidding

# Autocorrelation plot 

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

beta_cols = [c for c in samples.columns if c.startswith("beta")]

for col in beta_cols:
    plt.figure(figsize=(6,4))
    plot_acf(samples[col], lags=50)
    plt.title(f"ACF - {col}")
    plt.show()

plot_acf(samples["loglambda"], lags=50)
plt.title("ACF - loglambda")
plt.show()

az.summary(dataset_samples)

# Trace plot 

import matplotlib.pyplot as plt
import numpy as np

# 1. Identificar quantos draws existem por cadeia
# No PyStan 3, fit_pc.to_frame() empilha as cadeias sequencialmente.
total_linhas = samples.shape[0]
draws_por_cadeia = int(total_linhas / chains)

# 2. Separar as colunas do beta e mapear os nomes reais das variáveis
beta_cols = [c for c in samples.columns if c.startswith("beta")]
nomes_parametros = ["Intercept"] + list(Xs.columns)

# 3. Plotar os Betas
for col, nome_real in zip(beta_cols, nomes_parametros):
    plt.figure(figsize=(10, 3))
    
    # Cadeia 1 (Primeira metade dos dados)
    plt.plot(range(draws_por_cadeia), samples[col].iloc[:draws_por_cadeia], 
             label="Cadeia 1", color="#1f77b4", alpha=0.7, lw=0.6)
    
    # Cadeia 2 (Segunda metade dos dados)
    plt.plot(range(draws_por_cadeia), samples[col].iloc[draws_por_cadeia:], 
             label="Cadeia 2", color="#ff7f0e", alpha=0.7, lw=0.6)
    
    plt.title(f"Traceplot - {col} ({nome_real})")
    plt.xlabel("Iteração (Pós-Warmup)")
    plt.ylabel("Valor")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# 4. Plotar o loglambda
plt.figure(figsize=(10, 3))
plt.plot(range(draws_por_cadeia), samples["loglambda"].iloc[:draws_por_cadeia], 
         label="Cadeia 1", color="#1f77b4", alpha=0.7, lw=0.6)
plt.plot(range(draws_por_cadeia), samples["loglambda"].iloc[draws_por_cadeia:], 
         label="Cadeia 2", color="#ff7f0e", alpha=0.7, lw=0.6)

plt.title("Traceplot - loglambda")
plt.xlabel("Iteração (Pós-Warmup)")
plt.ylabel("Valor")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()



















