all:
	gcc ./include/parse_data.h ./source/parse_data.c -o parse

test_parse:
	gcc ./include/parse_data.h 
	gcc ./source/test_parse_data.c -o parse 	

headers:
	gcc ./include/parse_data.h
