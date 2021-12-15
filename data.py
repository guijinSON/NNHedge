def sequence_to_span(X_asset, X_option, span_length = 3):
    X_len = len(X_asset)
    asset_span = [X_asset[n:n+span_length] for n in range(X_len) if n < X_len - span_length +1 ] 
    span_option = X_option[span_length-1:]
    return asset_span,span_option

def generate_span_dataset(X_asset, X_option, span_length = 3):
    asset_span = []
    span_call =[]
    for Xa,Xo in zip(X_asset, X_option):
        data = sequence_to_span(Xa, Xo, span_length = span_length)
        asset_span.extend(data[0])
        span_call.extend(data[1])
    return asset_span, span_call
