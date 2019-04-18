default: bfs_frontier

bfs_frontier: bfs_frontier.cu 
	nvcc -O3 -arch=sm_35 -lineinfo bfs_frontier.cu -o bfs_frontier

clean:
	-rm -f bfs_frontier
