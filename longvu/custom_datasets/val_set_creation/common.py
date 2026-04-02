def format_ans_to_tvbench_style(ans: str, prefix_len: int):
    return (ans[prefix_len:].rstrip() + ".").capitalize()


def unformat_from_tvbench_style(ans: str):
    return ans.lower()[:-1]