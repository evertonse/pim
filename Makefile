DIR = main
PROJECT = pim
DIR_BIN = bin

push:
	git add . && git commit -m "PIM $$(date)${MSG}" && git push

pull:
	git pull

run: clean all
	./$(DIR_BIN)/$(PROJECT)

all:
	mkdir -p $(DIR_BIN)/
	odin build src -out:$(DIR_BIN)/$(PROJECT)
  

clean:
	rm -f $(DIR_BIN)/*.bin

.PHONY: all push run clean

