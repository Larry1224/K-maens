all: aos-ocl.exe soa-ocl.exe aos-omp.exe soa-omp.exe generator.exe

CC=g++
CFLAGS=-g -std=c++17 -march=native -fopenmp -lOpenCL

%.o: src/%.cpp 
	$(CC) $(CFLAGS) -c $< -o $@

%.exe: src/%.cpp
	$(CC) $(CFLAGS) $< -o $@

aos-omp.exe: src/main.cpp kmeansAOSp.o 
	$(CC) $(CFLAGS) $< -o $@ kmeansAOSp.o 

soa-omp.exe: src/main.cpp kmeansSOAp.o 
	$(CC) $(CFLAGS) $< -o $@ kmeansSOAp.o 

aos-ocl.exe: src/main.cpp kmeansAOSCL.o 
	$(CC) $< $(CFLAGS) -o $@ kmeansAOSCL.o 

soa-ocl.exe: src/main.cpp kmeansSOACL.o 
	$(CC) $< $(CFLAGS) -o $@ kmeansSOACL.o 
	
kmeans.exe: src/main.cpp kmeans.o 
	$(CC) $< $(CFLAGS) -o $@ kmeans.o
clean:
	rm *.exe *.o log

test: aos-ocl.exe 
	srun -w gpu0 ./aos-ocl.exe 2 0.000005 data/02_Skin_NonSkin.csv >>log
	# ./aos-ocl.exe 3 0.000005 data/02_Skin_NonSkin.csv >>log
	# ./aos-ocl.exe 3 0.000005 data/2-3-1000.csv >log2
	
	# valgrind --leak-check=full --show-leak-kinds=all --verbose ./aos-ocl.exe 3 0.000005 data/01_iris.csv >log
	# valgrind --leak-check=full --show-leak-kinds=all --verbose ./aos-ocl.exe 2 0.000005 data/02_Skin_NonSkin.csv >log

run: SHELL:=/bin/bash
run: aos-omp.exe soa-omp.exe aos-ocl.exe soa-ocl.exe
	for i in {1..100}; do \
		./aos-ocl.exe 3 0.000005 data/01_iris.csv >> AOS-01.csv; \
		./aos-ocl.exe 2 0.000005 data/02_Skin_NonSkin.csv >> AOS-02.csv; \
		./aos-omp.exe 3 0.000005 data/01_iris.csv >> AOS-01omp.csv; \
		./aos-omp.exe 2 0.000005 data/02_Skin_NonSkin.csv >> AOS-02omp.csv; \
		./soa-ocl.exe 3 0.000005 data/01_iris.csv >> SOA-01.csv; \
		./soa-ocl.exe 2 0.000005 data/02_Skin_NonSkin.csv >> SOA-02.csv; \
		./soa-omp.exe 3 0.000005 data/01_iris.csv >> SOA-01omp.csv; \
		./soa-omp.exe 2 0.000005 data/02_Skin_NonSkin.csv >> SOA-02omp.csv; \
	done

do: aos-omp.exe
	./aos-omp.exe 3 0.000005 data/02_Skin_NonSkin.csv >> log
	# ./aos-omp.exe 3 0.000005 data/02_Skin_NonSkin.csv >> log1
	# ./aos-omp.exe 3 0.000005 data/2-3-1000.csv >> log1

new: kmeans.exe
	srun -w gpu0 ./kmeans.exe 3 0.000005 data/01_iris.csv >>log

generate: generator.exe 
	./generator.exe 2 3 1000 > data/2-3-1000.csv
