"""Per-task class-macro metrics. No averaging over task heads is called macro-F1."""
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,roc_auc_score,log_loss

def classification_metrics(y,probabilities):
    y=np.asarray(y);p=np.asarray(probabilities);labels=np.arange(p.shape[1]);pred=p.argmax(1)
    result={"macro_f1":float(f1_score(y,pred,labels=labels,average="macro",zero_division=0)),
            "macro_precision":float(precision_score(y,pred,labels=labels,average="macro",zero_division=0)),
            "macro_recall":float(recall_score(y,pred,labels=labels,average="macro",zero_division=0)),
            "accuracy":float(np.mean(y==pred)),"confusion_matrix":confusion_matrix(y,pred,labels=labels).tolist(),
            "class_support":[int(np.sum(y==k)) for k in labels],"log_loss":float(log_loss(y,p,labels=labels))}
    if p.shape[1]==2 and len(np.unique(y))==2:result["auroc"]=float(roc_auc_score(y,p[:,1]))
    return result
