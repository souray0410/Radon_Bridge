"""Per-task class-macro metrics. No averaging over task heads is called macro-F1."""
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,roc_auc_score,log_loss

def binary_metrics(y, probabilities):
    """Preserve the native UKB binary metric schema without a training-repo import."""
    from sklearn.metrics import average_precision_score, brier_score_loss
    y = np.asarray(y)
    p = np.asarray(probabilities)
    return {
        "macro_f1": float(f1_score(y, p.argmax(1), labels=[0, 1], average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, p[:, 1])),
        "positive_average_precision": float(average_precision_score(y, p[:, 1])),
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p[:, 1])),
        "confusion_matrix": confusion_matrix(y, p.argmax(1), labels=[0, 1]).tolist(),
    }

def classification_metrics(y,probabilities):
    y=np.asarray(y);p=np.asarray(probabilities);labels=np.arange(p.shape[1]);pred=p.argmax(1)
    result={"macro_f1":float(f1_score(y,pred,labels=labels,average="macro",zero_division=0)),
            "macro_precision":float(precision_score(y,pred,labels=labels,average="macro",zero_division=0)),
            "macro_recall":float(recall_score(y,pred,labels=labels,average="macro",zero_division=0)),
            "accuracy":float(np.mean(y==pred)),"confusion_matrix":confusion_matrix(y,pred,labels=labels).tolist(),
            "class_support":[int(np.sum(y==k)) for k in labels],"log_loss":float(log_loss(y,p,labels=labels))}
    if p.shape[1]==2 and len(np.unique(y))==2:result["auroc"]=float(roc_auc_score(y,p[:,1]))
    return result


def batched_macro_f1(y, pred):
    """Binary macro-F1 over the last axis, with a fixed two-class label set."""
    values = []
    for c in [0, 1]:
        tp = ((pred == c) & (y == c)).sum(-1)
        den = (pred == c).sum(-1) + (y == c).sum(-1)
        values.append(np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0))
    return sum(values) / 2
