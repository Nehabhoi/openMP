/*
            Authour : Neha Bhoi
            Program : Simple code to optimized n X n matrix multiplication using OpenMp
*/

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

// assigned random value to matrix
void generate_matrix(int ** matrix, int s){
    for (int i = 0; i < s; i++) 
    {
        for (int j = 0; j < s; j++) 
        {
            matrix[i][j] = (rand() % 99) + 10;
        }
    }
}

// print matrix
void print_matrix(int ** matrix, int s){
    for(int i = 0; i< s; i++){
        printf("  | ");
        for(int j =0; j< s; j++){
            printf("%d\t", matrix[i][j]);
        }
        printf(" |\n");
    }
    printf("\n");
}

// serial matrix multiplication
void matrix_multiplication(int ** matrixA, int ** matrixB, int ** result, int s){
    for (int i = 0; i < s; i++) 
        {
            for (int j = 0; j < s; j++) 
            {
                result[i][j] = 0;
                for (int k = 0; k < s; k++) 
                {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }

    }
}

// paralle matrix multiplication
void matrix_multiplication_parallel(int ** matrixA, int ** matrixB, int ** result, int s){
    
    int i, j, k;
    #pragma omp parallel shared(matrixA,matrixB,result) private(i,j,k)
    {
        #pragma omp for schedule (static)
        for (i = 0; i < s; i++) 
        {
            for (j = 0; j < s; j++) 
            {
                result[i][j] = 0;
                for (k = 0; k < s; k++) 
                {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }

        }
    }
}

// checking for accuracy
void check_Accuracy(){
    int SIZE = 3;

    // create matrixA of (SIZE X SIZE) dimention and allocate memory to it
    int **matrixA = (int **) malloc(SIZE * sizeof(int *));
    for(int r = 0; r< SIZE; r++)
        matrixA[r] = (int *) malloc(SIZE * sizeof(int));
    generate_matrix(matrixA, SIZE);
    printf("  -------------------------------------------Input Matrix A: -------------------------------------------\n");
    print_matrix(matrixA, SIZE);

    // create matrixB of (SIZE X SIZE) dimention and allocate memory to it
    int **matrixB = (int **) malloc(SIZE * sizeof(int *));
    for(int r = 0; r< SIZE; r++)
        matrixB[r] = (int *) malloc(SIZE * sizeof(int));
    generate_matrix(matrixB, SIZE);
    printf("  -------------------------------------------Input Matrix B: -------------------------------------------\n");
    print_matrix(matrixB, SIZE);

    // create result_serial of (SIZE X SIZE) dimention and allocate memory to it
    int **result_serial = (int **) malloc(SIZE * sizeof(int *));
    for(int r = 0; r< SIZE; r++)
        result_serial[r] = (int *) malloc(SIZE * sizeof(int));
    //execute matrix multiplication using serial algorithm
    matrix_multiplication(matrixA,matrixB,result_serial,SIZE);
    printf("  -------------------------------------------Serial Matrix A * B: -------------------------------------------\n");
    print_matrix(result_serial, SIZE);

    // create result_parallel of (SIZE X SIZE) dimention and allocate memory to it
    int **result_parallel = (int **) malloc(SIZE * sizeof(int *));
    for(int r = 0; r< SIZE; r++)
        result_parallel[r] = (int *) malloc(SIZE * sizeof(int));
    //execute matrix muktiplication using parallel algorithm
    matrix_multiplication_parallel(matrixA,matrixB,result_parallel,SIZE);
    printf("  -------------------------------------------Parralel Matrix A * B: -------------------------------------------\n");
    print_matrix(result_parallel, SIZE);

    // deallocate memory
    for(int i =0 ; i<SIZE;i++){
        free(matrixA[i]);
        free(matrixB[i]);
        free(result_serial[i]);
        free(result_parallel[i]);
    }
    free(matrixA);
    free(matrixB);
    free(result_serial);
    free(result_parallel);
        
}

void execute_parallel_with_n_threads(int ** matrixA, int ** matrixB, int ** result_serial, int SIZE, int n, double time_to_execute_serial){
    double dtime;
    // create result_parallel of (SIZE X SIZE) dimention and allocate memory to it
    int **result_parallel = (int **) malloc(SIZE * sizeof(int *));
    for(int r = 0; r< SIZE; r++)
        result_parallel[r] = (int *) malloc(SIZE * sizeof(int));
    printf("  Number of threads: %d\n",n);
    omp_set_num_threads(n);
    dtime = omp_get_wtime();
    //execute matrix muktiplication using parallel algorithm with n threads
    matrix_multiplication_parallel(matrixA,matrixB,result_parallel,SIZE);
    double time_to_execute_with_n_threads = omp_get_wtime() - dtime;
    printf("  Time taken in parallel: %f\n",time_to_execute_with_n_threads);
    printf("  Speed Up: %f \n\n",time_to_execute_serial/time_to_execute_with_n_threads);

    //deallocate memory
    for(int i =0 ; i<SIZE;i++)
        free(result_parallel[i]);
    free(result_parallel);
}

void analyze_performance_based_on_matrix_size(int SIZE,int thread_num){
    // changing the size for higher value
    srand(time(NULL));
    double dtime;    
    printf("  -------------------------------------------New size of matrix is %d X %d-------------------------------------------\n\n",SIZE,SIZE);

    // create matrixA of (SIZE X SIZE) dimention and allocate memory to it
    int **matrixA = (int **) malloc(SIZE * sizeof(int *));
    for(int r = 0; r< SIZE; r++)
        matrixA[r] = (int *) malloc(SIZE * sizeof(int));
    generate_matrix(matrixA, SIZE);

    // create matrixB of (SIZE X SIZE) dimention and allocate memory to it
    int **matrixB = (int **) malloc(SIZE * sizeof(int *));
    for(int r = 0; r< SIZE; r++)
        matrixB[r] = (int *) malloc(SIZE * sizeof(int));
    generate_matrix(matrixB, SIZE);

    // create result_serial of (SIZE X SIZE) dimention and allocate memory to it
    int **result_serial = (int **) malloc(SIZE * sizeof(int *));
    for(int r = 0; r< SIZE; r++)
        result_serial[r] = (int *) malloc(SIZE * sizeof(int));
    dtime = omp_get_wtime();
    //execute matrix multiplication using serial algorithm
    matrix_multiplication(matrixA,matrixB,result_serial,SIZE);
    double time_to_execute_serial = omp_get_wtime() - dtime;
    printf("  Time taken in serial: %f\n\n",time_to_execute_serial);

    for(int i = 2; i<=thread_num;i = i+2){
        // execute matring multiplication using parallel algorithm with i threads
        execute_parallel_with_n_threads(matrixA, matrixB, result_serial, SIZE, i, time_to_execute_serial);
    }    

    //deallocate memory
    for(int i =0 ; i<SIZE;i++){
        free(matrixA[i]);
        free(matrixB[i]);
        free(result_serial[i]);
    }
    free(matrixA);
    free(matrixB);
    free(result_serial);
}

void main(){
    //check for Accuracy
    check_Accuracy();
    
    // system details
    printf ( "\n" );
    printf ( "  The number of processors available = %d\n", omp_get_num_procs ( ) );
    int thread_num  = omp_get_max_threads ( );
    printf ( "  The number of threads available    = %d\n",  omp_get_max_threads ( ));

    int SIZE = 250;
    analyze_performance_based_on_matrix_size(SIZE,thread_num);

    SIZE = 500;
    analyze_performance_based_on_matrix_size(SIZE,thread_num);

    SIZE = 1000;
    analyze_performance_based_on_matrix_size(SIZE,thread_num);

    SIZE = 2000;
    analyze_performance_based_on_matrix_size(SIZE,thread_num);
}