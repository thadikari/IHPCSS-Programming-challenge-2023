/**
 * @file main.f08
 * @brief This file provides you with the original implementation of pagerank.
 * Your challenge is to optimise it using OpenMP and/or MPI.
 * @author Ludovic Capelli (l.capelli@epcc.ed.ac.uk)
 **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>


/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed forthe calculation loop.
#define MAX_TIME 10

/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
int8_t adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
double inverse_outdegree[GRAPH_ORDER];
double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;
double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;


void initialize_graph(void)
{
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            adjacency_matrix[i][j] = 0.0;
        }
    }
}


void call_init_double_loop()
{
    for(int j = 0; j < GRAPH_ORDER; j++)
    {
        int outdegree = 0.;
        for(int k = 0; k < GRAPH_ORDER; k++)
        {
            outdegree += adjacency_matrix[j][k];
        }
        inverse_outdegree[j] = outdegree == 0.? 0. : 1./outdegree;
    }
}

/*
    Populate the L1 array with the number of non zero elements in each row of the adjacency matrix
*/
void get_l1_array(int L1[GRAPH_ORDER], int *l2_size)
{
    *l2_size = 0;
    for (int i=0; i<GRAPH_ORDER; i++)
    {
        int entries_per_row = 0;
        for (int j=0; j<GRAPH_ORDER; j++)
        {
            if (adjacency_matrix[j][i] != 0)
            {
                entries_per_row++;
            }
        }
        L1[i] = entries_per_row;
        *l2_size += entries_per_row;
    }
    return;
}

/*
    Populat the L2 array with the j indices of each non zero element of every row in the adjacency matrix
*/
void get_l2_array(const int l2_size, int L2[l2_size], int L3[GRAPH_ORDER])
{
    int offset = 0;
    for (int i=0; i<GRAPH_ORDER; i++)
    {
        L3[i] = offset;
        for (int j=0; j<GRAPH_ORDER; j++)
        {
            if (adjacency_matrix[j][i] != 0)
            {
                L2[offset] = j;
                offset++;
            }
        }
    }
    return;
}


/**
 * @brief Calculates the pagerank of all vertices in the graph.
 * @param pagerank The array in which store the final pageranks.
 */
void calculate_pagerank(double pagerank[])
{

    // Compute the L1 and L2 representation of the adjacency matrix
    int L1[GRAPH_ORDER];
    int L3[GRAPH_ORDER];
    int l2_size = 0;
    get_l1_array(L1, &l2_size);

    const int l2_size_const = l2_size;
    int L2[l2_size_const];
    get_l2_array(l2_size_const, L2, L3);

    double initial_rank = 1.0 / GRAPH_ORDER;

    // Initialise all vertices to 1/n.
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        pagerank[i] = initial_rank;
    }

    size_t iteration = 0;
    double start = omp_get_wtime();
    double elapsed = omp_get_wtime() - start;
    double time_per_iteration = 0;
    double new_pagerank[GRAPH_ORDER];

    // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X seconds on an iteration, and we are less than X seconds away from MAX_TIME, we stop.
    call_init_double_loop();

    int num_devices = omp_get_num_devices();
    printf("num_devices %d.\n", num_devices);

    double new_pagerank_2d[num_devices][GRAPH_ORDER];

    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        new_pagerank[i] = 0.0;
        int d = i%num_devices;
        new_pagerank_2d[d][i] = 0.0;
    }

    for (int d = 0; d < num_devices; d++)
    {
        #pragma omp target enter data device(d) map(alloc:L1[0:GRAPH_ORDER],L3[0:GRAPH_ORDER],L2[0:l2_size],adjacency_matrix[0:GRAPH_ORDER][0:GRAPH_ORDER],inverse_outdegree[0:GRAPH_ORDER],DAMPING_FACTOR,damping_value,pagerank[0:GRAPH_ORDER],new_pagerank_2d[d:1][0:GRAPH_ORDER])
        #pragma omp target update device(d) to(L1[0:GRAPH_ORDER],L3[0:GRAPH_ORDER],L2[0:l2_size],adjacency_matrix[0:GRAPH_ORDER][0:GRAPH_ORDER],inverse_outdegree[0:GRAPH_ORDER],DAMPING_FACTOR,damping_value,pagerank[0:GRAPH_ORDER],new_pagerank_2d[d:1][0:GRAPH_ORDER])
    }

    while(elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME)
    {
        double iteration_start = omp_get_wtime();
        double diff = 0.0;
        double pagerank_total = 0.0;

        #pragma omp parallel for
        for (int d = 0; d < num_devices; d++)
        {
            #pragma omp target update device(d) to(pagerank[0:GRAPH_ORDER])

            #pragma omp target teams distribute device(d)
            for(int i = d; i < GRAPH_ORDER; i += num_devices)
            {
                int offset = L3[i];
                const int nonzero_per_row = L1[i];
                double total = 0.;
                #pragma omp parallel for default(none) shared(i,L2,offset,adjacency_matrix,pagerank,inverse_outdegree) reduction(+:total)
                for(int l = 0; l < nonzero_per_row; l++)
                {
                    int j = L2[offset+l];
                    total += adjacency_matrix[j][i] * pagerank[j] * inverse_outdegree[j];
                }
                new_pagerank_2d[d][i] = DAMPING_FACTOR * total + damping_value;
            }

            #pragma omp target update device(d) from(new_pagerank_2d[d:1][0:GRAPH_ORDER])
        }

        #pragma omp parallel for reduction(+:diff) reduction(+:pagerank_total)
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            int d = i%num_devices;
            new_pagerank[i] = new_pagerank_2d[d][i];

            diff += fabs(new_pagerank[i] - pagerank[i]);
            pagerank[i] = new_pagerank[i];
            pagerank_total += pagerank[i];
        }

        max_diff = (max_diff < diff) ? diff : max_diff;
        total_diff += diff;
        min_diff = (min_diff > diff) ? diff : min_diff;

        if(fabs(pagerank_total - 1.0) >= 1.0)
        {
            printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n", iteration, pagerank_total);
        }

		double iteration_end = omp_get_wtime();
		elapsed = omp_get_wtime() - start;
		iteration++;
		time_per_iteration = elapsed / iteration;
    }

    for (int d = 0; d < num_devices; d++)
    {
        // #pragma omp target exit data device(d) map(delete:L1,L3,L2[0:l2_size],pagerank,new_pagerank,adjacency_matrix,inverse_outdegree,DAMPING_FACTOR,damping_value)
    }

    printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
}

/**
 * @brief Populates the edges in the graph for testing.
 **/
void generate_nice_graph(void)
{
    printf("Generate a graph for testing purposes (i.e.: a nice and conveniently designed graph :) )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            int source = i;
            int destination = j;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

/**
 * @brief Populates the edges in the graph for the challenge.
 **/
void generate_sneaky_graph(void)
{
    printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER - i; j++)
        {
            int source = i;
            int destination = j;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

int main(int argc, char* argv[])
{
    // We do not need argc, this line silences potential compilation warnings.
    (void) argc;
    // We do not need argv, this line silences potential compilation warnings.
    (void) argv;

    printf("This program has two graph generators: generate_nice_graph and generate_sneaky_graph. If you intend to submit, your code will be timed on the sneaky graph, remember to try both.\n");

    // Get the time at the very start.
    double start = omp_get_wtime();

    generate_sneaky_graph();

    /// The array in which each vertex pagerank is stored.
    double pagerank[GRAPH_ORDER];
    calculate_pagerank(pagerank);

    // Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
    double sum_ranks = 0.0;
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        if(i % 100 == 0)
        {
            printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
        }
        sum_ranks += pagerank[i];
    }
    printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", sum_ranks, total_diff, max_diff, min_diff);
    double end = omp_get_wtime();

    printf("Total time taken: %.2f seconds.\n", end - start);

    return 0;
}
