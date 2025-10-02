# %% [markdown]
# # Спам-фильтр на выборке UCI Spambase
# Нотебук: анализ данных, препроцессинг, эксперименты с Перцептроном и Логистической регрессией, подбор гиперпараметров, сравнение итоговых моделей.

# %%
# Библиотеки и настройки
import os, io, zipfile, warnings, textwrap, itertools, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    cross_validate, GridSearchCV, validation_curve, learning_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.utils import Bunch

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## Загрузка датасета
# * Предпочтительно читаем локальные файлы (`spambase.data`, `spambase.names`/`DOCUMENTATION`).
# * Если их нет — пробуем скачать с UCI.

# %%
# Имена признаков из документации UCI (57 признаков + целевая переменная 'spam')
word_freq = [
    "make","address","all","3d","our","over","remove","internet","order","mail","receive",
    "will","people","report","addresses","free","business","email","you","credit","your",
    "font","000","money","hp","hpl","george","650","lab","labs","telnet","857","data","415",
    "85","technology","1999","parts","pm","direct","cs","meeting","original","project","re",
    "edu","table","conference"
]
char_freq = [";","(", "[","!","$","#"]
char_freq = [f"char_freq_{c}" for c in ["semicolon","lbracket","lparen","exclam","dollar","hash"]]
capital_feats = ["capital_run_length_average","capital_run_length_longest","capital_run_length_total"]
SPAMBASE_COLUMNS = [f"word_freq_{w}" for w in word_freq] + char_freq + capital_feats + ["spam"]

def load_spambase(path: str = ".") -> pd.DataFrame:
    # 1) прямой файл
    for fname in ["spambase.data", "spambase.csv"]:
        f = os.path.join(path, fname)
        if os.path.exists(f):
            df = pd.read_csv(f, header=None, names=SPAMBASE_COLUMNS)
            return df
    # 2) zip-архив
    zf = os.path.join(path, "spambase.zip")
    if os.path.exists(zf):
        with zipfile.ZipFile(zf, "r") as z:
            inner = [n for n in z.namelist() if n.endswith("spambase.data")]
            if inner:
                with z.open(inner[0]) as f:
                    df = pd.read_csv(f, header=None, names=SPAMBASE_COLUMNS)
                    return df
    # 3) загрузка с UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    try:
        df = pd.read_csv(url, header=None, names=SPAMBASE_COLUMNS)
        return df
    except Exception as e:
        raise RuntimeError("Не удалось найти ни локальные файлы, ни загрузить с UCI.") from e

df = load_spambase()
df.head()

# %% [markdown]
# ## Базовый анализ
# Размер датасета, типы, пропуски, баланс классов, описательная статистика.

# %%
print("Размер:", df.shape)
print(df.dtypes.value_counts())
print("Число пропусков всего:", int(df.isna().sum().sum()))
print("Баланс классов (0=ham, 1=spam):")
print(df["spam"].value_counts().rename("count").to_frame().assign(frac=lambda s: s["count"]/len(df)))

print(df.describe().T.round(3).iloc[:10])  # первые 10 признаков для компактности

# %%
# Корреляции и наиболее коррелирующие признаки с целевой
corr = df.corr(numeric_only=True)
top_to_spam = corr["spam"].sort_values(key=np.abs, ascending=False).head(15)
print(top_to_spam.to_frame("corr_with_spam").round(3))

plt.figure(figsize=(10,6))
sns.barplot(x=top_to_spam.index, y=top_to_spam.values)
plt.xticks(rotation=60, ha="right")
plt.title("Наиболее коррелирующие признаки с целевой")
plt.tight_layout()
plt.show()

# %%
# Топ распределений для нескольких признаков + сравнение классов
cols_demo = ["word_freq_free","word_freq_your","word_freq_money","char_freq_exclam",
             "capital_run_length_average","capital_run_length_longest"]
melted = df[cols_demo+["spam"]].melt("spam", var_name="feature", value_name="value")
g = sns.FacetGrid(melted, col="feature", col_wrap=3, height=3, sharex=False, sharey=False, hue="spam")
g.map(sns.kdeplot, "value", common_norm=False, fill=True, alpha=0.4)
g.add_legend(title="spam")
plt.suptitle("Плотности распределений по классам", y=1.03)
plt.show()

# %%
# Быстрый 2D-обзор: PCA
from sklearn.decomposition import PCA
X_num = df.drop(columns=["spam"]).values
y = df["spam"].values
X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_num)

plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=12, alpha=0.6, cmap="coolwarm")
plt.title("PCA (2 компоненты)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Препроцессинг
# * Категориальные → OneHot (на всякий случай, хотя в Spambase их нет).
# * Числовые → StandardScaler.

# %%
# Выделение типов признаков
feature_cols = [c for c in df.columns if c!="spam"]
cat_cols = [c for c in feature_cols if df[c].dtype=="object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

X = df[feature_cols]
y = df["spam"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}, Spam rate train: {y_train.mean():.3f}")

# %% [markdown]
# ## Базовые модели (без подбора)
# Стартовые оценки по 5-крж кросс-валидации.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"roc_auc":"roc_auc", "f1":"f1", "bal_acc":"balanced_accuracy", "ap":"average_precision"}

def evaluate_cv(pipe, X, y, name):
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    row = {f"cv_{k}":res[f"test_{k}"].mean() for k in scoring}
    row.update({f"std_{k}":res[f"test_{k}"].std() for k in scoring})
    row["model"] = name
    return row

pipe_lr_base = Pipeline([("prep", preprocess),
                         ("clf", LogisticRegression(max_iter=500, solver="saga", random_state=RANDOM_STATE))])

pipe_pc_base = Pipeline([("prep", preprocess),
                         ("clf", Perceptron(random_state=RANDOM_STATE))])

rows = []
rows.append(evaluate_cv(pipe_lr_base, X, y, "LogReg base"))
rows.append(evaluate_cv(pipe_pc_base, X, y, "Perceptron base"))
cv_table = pd.DataFrame(rows).set_index("model").sort_values("cv_roc_auc", ascending=False)
print(cv_table.round(4))

# %% [markdown]
# ## Подбор гиперпараметров
# Гриды и много-метрическая оптимизация.

# %%
# Логистическая регрессия: C, penalty, l1_ratio, class_weight
pipe_lr = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(solver="saga", max_iter=5000, random_state=RANDOM_STATE))
])

grid_lr = [
    {"clf__penalty": ["l2"],
     "clf__C": np.logspace(-3, 3, 13),
     "clf__class_weight": [None, "balanced"]},
    {"clf__penalty": ["l1"],
     "clf__C": np.logspace(-3, 3, 13),
     "clf__class_weight": [None, "balanced"]},
    {"clf__penalty": ["elasticnet"],
     "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
     "clf__C": np.logspace(-3, 3, 13),
     "clf__class_weight": [None, "balanced"]},
]

gs_lr = GridSearchCV(
    pipe_lr, grid_lr, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0, refit=True
)
gs_lr.fit(X_train, y_train)

print("Best LR params:", gs_lr.best_params_)
print("Best CV ROC-AUC:", gs_lr.best_score_)

# %%
# Перцептрон: penalty, alpha, l1_ratio, class_weight, early_stopping и max_iter
pipe_pc = Pipeline([
    ("prep", preprocess),
    ("clf", Perceptron(random_state=RANDOM_STATE))
])

grid_pc = {
    "clf__penalty": [None, "l2", "l1", "elasticnet"],
    "clf__alpha": np.logspace(-6, -1, 6),
    "clf__l1_ratio": [0.0, 0.15, 0.5, 0.85],
    "clf__class_weight": [None, "balanced"],
    "clf__early_stopping": [True],
    "clf__validation_fraction": [0.1, 0.2],
    "clf__max_iter": [2000, 4000],
    "clf__eta0": [0.1, 1.0],
    "clf__shuffle": [True]
}

gs_pc = GridSearchCV(
    pipe_pc, grid_pc, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0, refit=True
)
gs_pc.fit(X_train, y_train)

print("Best Perceptron params:", gs_pc.best_params_)
print("Best CV ROC-AUC:", gs_pc.best_score_)

# %% [markdown]
# ## Итоговая оценка на отложенном тесте

# %%
def evaluate_on_test(estimator, X_train, y_train, X_test, y_test, name="model"):
    est = estimator
    y_proba = est.predict_proba(X_test)[:,1] if hasattr(est, "predict_proba") else None
    y_dec = est.decision_function(X_test) if hasattr(est, "decision_function") else None
    y_pred = est.predict(X_test)

    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["ham","spam"])
    disp.plot(values_format="d"); plt.title(f"Confusion Matrix: {name}"); plt.show()

    if y_proba is None and y_dec is not None:
        # приведем к [0,1] монотонным преобразованием для графиков
        y_proba = (y_dec - y_dec.min()) / (y_dec.max() - y_dec.min() + 1e-12)

    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)
        fpr,tpr,_ = roc_curve(y_test, y_proba)
        pr, rc, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.4f}")
        plt.plot([0,1],[0,1],"--", lw=1)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC: {name}"); plt.legend(); plt.tight_layout(); plt.show()

        plt.figure(figsize=(5,4))
        plt.plot(rc, pr, label=f"AP={ap:.4f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR: {name}"); plt.legend(); plt.tight_layout(); plt.show()

        return {"roc_auc": roc, "ap": ap}
    else:
        return {}

best_lr = gs_lr.best_estimator_
best_pc = gs_pc.best_estimator_

metrics_lr = evaluate_on_test(best_lr, X_train, y_train, X_test, y_test, "LogisticRegression (best)")
metrics_pc = evaluate_on_test(best_pc, X_train, y_train, X_test, y_test, "Perceptron (best)")

print("Test metrics:")
print(pd.DataFrame([{"model":"LogReg","roc_auc":metrics_lr.get("roc_auc"),"ap":metrics_lr.get("ap")},
                      {"model":"Perceptron","roc_auc":metrics_pc.get("roc_auc"),"ap":metrics_pc.get("ap")}]).set_index("model").round(4))

# %% [markdown]
# ## Валидационные кривые для регуляризации
# Как ведет себя качество при разных C (LogReg) и alpha (Perceptron).

# %%
def plot_val_curve(pipe, X, y, param_name, param_range, scoring="roc_auc", logx=True, title=None):
    train_scores, test_scores = validation_curve(
        pipe, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    tr_mean, te_mean = train_scores.mean(axis=1), test_scores.mean(axis=1)
    plt.figure(figsize=(6,4))
    if logx: 
        plt.semilogx(param_range, tr_mean, label="train")
        plt.semilogx(param_range, te_mean, label="cv")
    else:
        plt.plot(param_range, tr_mean, label="train")
        plt.plot(param_range, te_mean, label="cv")
    plt.xlabel(param_name); plt.ylabel(scoring); 
    plt.title(title or f"Validation curve: {param_name}")
    plt.legend(); plt.tight_layout(); plt.show()

pipe_lr_l2 = Pipeline([("prep", preprocess),
                       ("clf", LogisticRegression(penalty="l2", solver="saga", max_iter=5000, random_state=RANDOM_STATE))])

plot_val_curve(pipe_lr_l2, X, y, "clf__C", np.logspace(-3, 3, 13), "roc_auc", True, "LogReg (L2) — влияние C")

pipe_pc_el = Pipeline([("prep", preprocess),
                       ("clf", Perceptron(penalty="elasticnet", random_state=RANDOM_STATE, early_stopping=True))])

plot_val_curve(pipe_pc_el, X, y, "clf__alpha", np.logspace(-6, -1, 10), "roc_auc", True, "Perceptron (elasticnet) — влияние alpha")

# %% [markdown]
# ## Кривые обучения
# Смотрим переобучение/недообучение.

# %%
def plot_learning_curve(pipe, X, y, title):
    sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=cv, n_jobs=-1, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True, random_state=RANDOM_STATE
    )
    plt.figure(figsize=(6,4))
    plt.plot(sizes, train_scores.mean(axis=1), label="train")
    plt.plot(sizes, test_scores.mean(axis=1), label="cv")
    plt.xlabel("Train size"); plt.ylabel("ROC-AUC"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()

plot_learning_curve(best_lr, X, y, "Learning Curve — Logistic Regression (best)")
plot_learning_curve(best_pc, X, y, "Learning Curve — Perceptron (best)")

# %% [markdown]
# ## Интерпретация признаков для LogReg
# Топ-коэффициенты (по абсолютной величине).

# %%
# Восстановим имена фич после препроцессинга
prep = best_lr.named_steps["prep"]
feat_names = prep.get_feature_names_out() if hasattr(prep, "get_feature_names_out") else np.array(feature_cols)

clf_lr = best_lr.named_steps["clf"]
coefs = clf_lr.coef_.ravel()
coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs, "abs": np.abs(coefs)}).sort_values("abs", ascending=False)

top_k = 20
fig, ax = plt.subplots(figsize=(7,6))
sns.barplot(data=coef_df.head(top_k), x="abs", y="feature", hue=(coef_df.head(top_k)["coef"]>0).map({True:"+",False:"-"}), dodge=False)
ax.set_title("Топ весов логистической регрессии")
ax.set_xlabel("|коэффициент|"); ax.set_ylabel("feature"); ax.legend(title="знак"); plt.tight_layout(); plt.show()

print(coef_df.head(30).drop(columns="abs").round(4))

# %% [markdown]
# ## Сравнение моделей и вывод
# Печатаем сводную таблицу и краткий текстовый вывод.

# %%
summary_rows = []

def holdout_metrics(est, name):
    y_proba = est.predict_proba(X_test)[:,1] if hasattr(est, "predict_proba") else None
    if y_proba is None and hasattr(est, "decision_function"):
        s = est.decision_function(X_test)
        y_proba = (s - s.min())/(s.max()-s.min()+1e-12)
    y_pred = est.predict(X_test)
    row = dict(
        model=name,
        roc_auc=roc_auc_score(y_test, y_proba),
        ap=average_precision_score(y_test, y_proba),
        bal_acc=(confusion_matrix(y_test, y_pred, normalize="true").diagonal().mean())
    )
    return row

summary_rows.append(holdout_metrics(best_lr, "LogReg (best)"))
summary_rows.append(holdout_metrics(best_pc, "Perceptron (best)"))

summary = pd.DataFrame(summary_rows).set_index("model").round(4)
print(summary)

# Краткий автоматический вывод о победителе
best_name = summary["roc_auc"].idxmax()
print(f"Лучшая модель по ROC-AUC на тесте: {best_name}")
print("Комментарий:")
print("* Логистическая регрессия обычно выигрывает на линейно разделимых задачах с шумом и хорошо работает с L1/L2/elasticnet регуляризацией.\n"
      "* Перцептрон чувствителен к масштабу и шуму, при этом не выдает вероятности из коробки. В этой задаче он уступает по PR/ROC.")

# %% [markdown]
# ## Доп. эксперименты: сравнение разных регуляризаторов для LogReg
# Трассируем средний CV ROC-AUC по `C` для L1/L2/ElasticNet.

# %%
def trace_lr_cv(penalty, l1_ratio=None, C_grid=np.logspace(-3,3,13)):
    scores = []
    for C in C_grid:
        lr = Pipeline([("prep", preprocess),
                       ("clf", LogisticRegression(
                           solver="saga", penalty=penalty, l1_ratio=l1_ratio,
                           C=C, max_iter=5000, random_state=RANDOM_STATE
                       ))])
        cv_res = cross_validate(lr, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        scores.append(cv_res["test_score"].mean())
    return C_grid, np.array(scores)

C_grid = np.logspace(-3,3,13)
curves = []
for pen in ["l1","l2"]:
    c, s = trace_lr_cv(penalty=pen, C_grid=C_grid)
    curves.append((f"LR-{pen.upper()}", c, s))
for l1r in [0.1,0.5,0.9]:
    c, s = trace_lr_cv(penalty="elasticnet", l1_ratio=l1r, C_grid=C_grid)
    curves.append((f"LR-EN(l1_ratio={l1r})", c, s))

plt.figure(figsize=(7,5))
for name, c, s in curves:
    plt.semilogx(c, s, label=name)
plt.xlabel("C"); plt.ylabel("CV ROC-AUC"); plt.title("Регуляризация в Logistic Regression")
plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Сохранение лучших моделей (при желании)
# Модели сохраняются как pickle-файлы.

# %%
import joblib, datetime, pathlib
outdir = pathlib.Path("models")
outdir.mkdir(exist_ok=True)
joblib.dump(best_lr, outdir / "logreg_best.pkl")
joblib.dump(best_pc, outdir / "perceptron_best.pkl")
print("Сохранено в:", outdir.resolve())
