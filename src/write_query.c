#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#define DATA_MIN -6.00
#define DATA_MAX 6.00
#define CAT_MIN -1.0
#define CAT_MAX 1.0
#define TIME_L_MIN -3.0
#define TIME_L_MAX 3.0
// TIME_R >= TIME_L
#define TIME_R_MAX 4.0

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));
    int fd = open(argv[1], O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
    uint32_t number = strtoul(argv[2], NULL, 0);
    ftruncate(fd, (number*(sizeof(float)*104))+sizeof(uint32_t));
    write(fd, &number, 4);
    float *buffer = malloc(104*sizeof(float));
    if(!buffer) {
        printf("Could not allocate buffer!\n");
        exit(1);
    }
    for(int i = 0; i < number; i++) {
        // Query Type
        int qt = (4 * ((float)rand() / RAND_MAX));
        buffer[0] = (float)qt;
        switch(qt) {
            case 3: buffer[1] = ((CAT_MAX - CAT_MIN) * ((float)rand() / RAND_MAX)) + CAT_MIN;
                    buffer[2] = ((TIME_L_MAX - TIME_L_MIN) * ((float)rand() / RAND_MAX)) + TIME_L_MIN;
                    buffer[3] = ((TIME_R_MAX - buffer[2]) * ((float)rand() / RAND_MAX)) + buffer[2];
                    break;
            case 2: buffer[1] = -1.0;
                    buffer[2] = ((TIME_L_MAX - TIME_L_MIN) * ((float)rand() / RAND_MAX)) + TIME_L_MIN;
                    buffer[3] = ((TIME_R_MAX - buffer[2]) * ((float)rand() / RAND_MAX)) + buffer[2];
                    break;
            case 1: buffer[1] = ((CAT_MAX - CAT_MIN) * ((float)rand() / RAND_MAX)) + CAT_MIN;
                    buffer[2] = -1.0;
                    buffer[3] = -1.0;
                    break;
            case 0: buffer[1] = -1.0;
                    buffer[2] = -1.0;
                    buffer[3] = -1.0;
                    break;
            default: printf("Something went wrong...\n"); goto end;
        }
        for(int j = 4; j < 104; j++) {
            buffer[j] = ((DATA_MAX - DATA_MIN) * ((float)rand() / RAND_MAX)) + DATA_MIN;
        }
        if(write(fd, buffer, 104*sizeof(float)) == -1) {
            printf("Something went wrong while writing the file!\n");
            break;
        }
    }
    end:
    fsync(fd);
    close(fd);
    free(buffer);
    return 0;
}
