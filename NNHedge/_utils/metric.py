def entropic_loss(pnl):
    pnl = torch.tensor(pnl)
    return -torch.mean(-torch.exp(-pnl)).numpy()

def evaluate_model(Y):
    Y = pd.Series(Y)

    metric = {
        "Entropic Loss Measure (ERM)" : entropic_loss(Y),
        "Mean" : Y.mean(),
        "VaR99" : Y.quantile(0.01),
        "VaR95" : Y.quantile(0.05),
        "VaR90" : Y.quantile(0.1),
        "VaR80" : Y.quantile(0.2),
        "VaR50" : Y.quantile(0.5)
    }
    return metric
