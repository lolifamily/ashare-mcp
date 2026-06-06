from baostock.data.resultset import ResultData

def query_history_k_data_plus(
    code: str,
    fields: str,
    start_date: str | None = ...,
    end_date: str | None = ...,
    frequency: str = ...,
    adjustflag: str = ...,
) -> ResultData: ...
