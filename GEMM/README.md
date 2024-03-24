### learning to Implement and optimize GEMM

### how to use
```bash
1. clang-format -i gemm.cu

// run one cuda kernel
1. nvcc -std=c++11 -lcublas -o a gemm.cu
2. ./a 4096 4096 4096

// run all cuda kernel
1. make

// docker run
1. docker pull nvidia/cuda:12.2.2-devel-ubuntu20.04
2. docker run --runtime=nvidia -it --privileged=true --name test4 nvidia/cuda:12.2.2-devel-ubuntu20.04 /bin/bash --gpus 'device=1' /bin/bash
2. docker cp GEMM/ f91879a9950e:/workspace/
3. nvcc -std=c++11 -lcublas -o a gemm.cu
4. ncu -o profile_tmp_golden -f --set full --graph-profiling=graph ./a 4096 4096 4096
5. docker cp f91879a9950e:/workspace/GEMM/profile_tmp_golden.ncu-rep ./tmp_golden.ncu-rep


docker ps // live docker
docker images // all docker image
docker start docker_name
docker exec -it docker_name /bin/bash
docker stop docker_name
```

![GEMM](GEMM.jpg)


### Referecnce
1. https://zhuanlan.zhihu.com/p/435908830
2. https://zhuanlan.zhihu.com/p/442930482
3. https://zhuanlan.zhihu.com/p/481600052
4. https://zhuanlan.zhihu.com/p/614109686