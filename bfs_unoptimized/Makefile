default: bfs_unoptimized

bfs_unoptimized: bfs_unoptimized.cu 
	nvcc -O3 -arch=sm_35 -lineinfo bfs_unoptimized.cu -o bfs_unoptimized

clean:
	-rm -f bfs_unoptimized
