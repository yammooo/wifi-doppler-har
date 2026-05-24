from dataset import parse_trace_filename


if __name__ == "__main__":
    examples = [
        "S1a_W_1_stream_0.txt",
        "S1a_J1_stream_2.txt",
        "S1a_E_stream_3.txt",
        "PI1a_p03_stream_0.txt",
    ]

    for example in examples:
        print(example, "->", parse_trace_filename(example))
