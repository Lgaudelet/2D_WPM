# Variables

SRC=src/main.cc  src/system.cc  src/wpm.cc src/kernel.cu 
OBJ=main.o  system.o  wpm.o  kernel.o

CC=nvcc
CFLAGS=-O2 -std=c++11

INC=-I./inc 
LIB=-L./lib -lcufft 
LDFLAGS=$(INC) $(LIB)

# all
.PHONY: all
all: ./bin/wpm

%.o: src/%.cc
	$(CC) $(LDFLAGS) $(CFLAGS) -c -o $@ $<

kernel.o : src/kernel.cu
	$(CC) $(LDFLAGS) $(CFLAGS) -c -o $@ $<

./bin/wpm: $(OBJ)
	$(CC) $(LDFLAGS) $(CFLAGS)  -o $@ $^


# clean
.PHONY: clean
clean:
	rm -f *.o

.PHONY: mrproper
mrproper:
	rm -f *.o *.csv ./bin/wpm log.txt


###
# profiling
KERNEL=0
#METRICS=sm_efficiency,branch_efficiency,warp_execution_efficiency
EVENTS=divergent_branch,shared_ld_bank_conflict,shared_st_bank_conflict,shared_ld_transactions,shared_st_transactions,l1_shared_bank_conflict
SIZE=256

profile:
	#srun --gres=gpu nvprof --metrics $(METRICS) --events $(EVENTS) ./bin/wpm -k $(KERNEL) -x $(SIZE) 
	srun --gres=gpu nvprof --events $(EVENTS) ./bin/wpm -k $(KERNEL) -x $(SIZE) 




