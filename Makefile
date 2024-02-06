CC = gcc
SDL2_CFLAGS := $(shell pkg-config --cflags sdl2 SDL2_image)
SDL2_LIBS := $(shell pkg-config --libs sdl2 SDL2_image)
CPPFLAGS = -I./include $(SDL2_CFLAGS)
CFLAGS = -Wall -Wextra -fsanitize=address
LDFLAGS = -fsanitize=address $(SDL2_LIBS)
LDLIBS = -lm
BUILD_DIR = ./build

.PHONY: all clean directories ocr neural 

all: ocr

ocr: directories $(BUILD_DIR)/ocr

neural: ocr

directories: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/temp

$(BUILD_DIR)/ocr: src/ocr.c src/neuralnetwork.c src/idx.c src/convert.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@cp src/trained/digits.neu $(BUILD_DIR)/temp

clean:
	rm -rf $(BUILD_DIR)
