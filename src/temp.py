import re

PATTERN = re.compile(
    r"^(?P<scenario>S(?P<scenario_id>\d+)(?P<campaign>[a-z]))_(?P<activity>[A-Z])(?P<repetition>\d*)_stream_(?P<antenna>\d+)\.txt$"
)

m = PATTERN.match("S1a_J1_stream_3.txt")
print(m.groupdict())