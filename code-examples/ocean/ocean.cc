
//int W = 1000+1;
//int H = 1000+1;
#define W (10+1)
#define H (10+1)

void sequentialOceanIteration(float in[W][H], float out[W][H]) {
	for (int i=0; i < W; ++i) {
		for (int j=0; j < H; ++j) {
			out[i][j] = 0.2*(in[i][j]+in[i-1][j]+
					in[i][j-1]+in[i+1][j]+
					in[i][j+1]);
		}
	}
}



int main ()
{
    int ITERS=1;
    float in[W][H];
    float out[W][H];
    for (int i = 0; i < ITERS; ++i) {
        sequentialOceanIteration(in,out);
    }

    return 0;
}
