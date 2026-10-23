# Builds the C example against the release libkapel of this workspace:
#
#     cargo build --release -p kapel-capi
#     make -C crates/kapel-capi/examples/c
#     crates/kapel-capi/examples/c/generate HuggingFaceTB/SmolLM2-135M-Instruct "Hello"

TARGET ?= $(abspath ../../../../target/release)
CFLAGS ?= -std=c99 -Wall -Wextra -O2

generate: generate.c ../../include/kapel.h
	$(CC) $(CFLAGS) generate.c -I ../../include -L $(TARGET) -lkapel -Wl,-rpath,$(TARGET) -o $@

clean:
	rm -f generate

.PHONY: clean
