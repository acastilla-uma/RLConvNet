#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog);
static int parse_int_arg(const char *opt, const char *value, int *out);
static int parse_double_arg(const char *opt, const char *value, double *out);

#define W 100
#define H 100
#define MAX_ORIENT 8
#define MAX_ACTIONS 6
#define DEFAULT_K_ITERS 1000
#define DEFAULT_ALLOWED_ACTIONS 3

static int n_orient = MAX_ORIENT;
static int n_actions = MAX_ACTIONS;
static int n_allowed_actions = DEFAULT_ALLOWED_ACTIONS;
static int k_max = DEFAULT_K_ITERS;
static double tol = 1e-4;

static double kernel[MAX_ORIENT][MAX_ACTIONS][3][3];
static int gmap[MAX_ORIENT][MAX_ACTIONS];

static int action_allowed(int action) {
    return action < n_allowed_actions;
}

static int parse_int_arg(const char *opt, const char *value, int *out) {
    if (!value) {
        fprintf(stderr, "%s expects a value\n", opt);
        return -1;
    }
    char *end = NULL;
    long v = strtol(value, &end, 10);
    if (end == value) {
        fprintf(stderr, "Invalid integer for %s: %s\n", opt, value);
        return -1;
    }
    *out = (int)v;
    return 0;
}

static int parse_double_arg(const char *opt, const char *value, double *out) {
    if (!value) {
        fprintf(stderr, "%s expects a value\n", opt);
        return -1;
    }
    char *end = NULL;
    double v = strtod(value, &end);
    if (end == value) {
        fprintf(stderr, "Invalid float for %s: %s\n", opt, value);
        return -1;
    }
    *out = v;
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options] [reward.csv]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --reward <path>          Path to reward CSV (positional arg also accepted)\n");
    fprintf(stderr, "  --k-max <int>            Maximum value-iteration steps (default %d)\n", DEFAULT_K_ITERS);
    fprintf(stderr, "  --tol <float>            Convergence tolerance (default 1e-4)\n");
    fprintf(stderr, "  --n-orient <int>         Number of orientations (1-%d, default %d)\n", MAX_ORIENT, MAX_ORIENT);
    fprintf(stderr, "  --n-actions <int>        Number of actions per orientation (1-%d, default %d)\n", MAX_ACTIONS, MAX_ACTIONS);
    fprintf(stderr, "  --allowed-actions <int>  Effective action budget (default %d)\n", DEFAULT_ALLOWED_ACTIONS);
    fprintf(stderr, "  --help                   Show this message\n");
}

static int load_reward_csv(const char *path, double reward[H][W]) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open reward CSV: %s\n", path);
        return 1;
    }

    char line[8192];
    int row = 0;
    while (fgets(line, (int)sizeof(line), fp)) {
        char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            ++p;
        }
        if (*p == '\0') {
            continue;
        }
        if (row >= H) {
            fprintf(stderr, "CSV has more than %d rows\n", H);
            fclose(fp);
            return 1;
        }

        for (int col = 0; col < W; ++col) {
            char *end = NULL;
            double v = strtod(p, &end);
            if (end == p) {
                fprintf(stderr, "CSV parse error at row %d, col %d\n", row, col);
                fclose(fp);
                return 1;
            }
            reward[row][col] = v;
            p = end;
            while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
                ++p;
            }
        }

        char *end = NULL;
        (void)strtod(p, &end);
        if (end != p) {
            fprintf(stderr, "CSV has more than %d columns at row %d\n", W, row);
            fclose(fp);
            return 1;
        }

        ++row;
    }

    fclose(fp);
    if (row != H) {
        fprintf(stderr, "CSV row count mismatch: expected %d, got %d\n", H, row);
        return 1;
    }

    return 0;
}

static char* extract_map_folder(const char *reward_path) {
    // Extract folder path from reward.csv path
    // e.g., "sim_maps/map_d0.35_s42_sm3_cr2/reward.csv" -> "sim_maps/map_d0.35_s42_sm3_cr2"
    static char folder[512];
    strncpy(folder, reward_path, sizeof(folder) - 1);
    folder[sizeof(folder) - 1] = '\0';
    
    // Find and remove the filename part
    char *last_slash = strrchr(folder, '/');
    if (!last_slash) {
        last_slash = strrchr(folder, '\\');
    }
    if (last_slash) {
        *last_slash = '\0';
    }
    
    return folder;
}

static int write_policy_txt(const char *path, double policy[MAX_ORIENT][MAX_ACTIONS][H][W], int iters_used) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open policy output: %s\n", path);
        return 1;
    }

    fprintf(fp, "# iters=%d n_orient=%d n_actions=%d allowed_actions=%d\n", iters_used, n_orient, n_actions, n_allowed_actions);

    for (int j = 0; j < n_orient; ++j) {
        for (int i = 0; i < n_actions; ++i) {
            fprintf(fp, "# orient=%d action=%d\n", j, i);
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    fprintf(fp, "%.10f", policy[j][i][y][x]);
                    if (x + 1 < W) {
                        fputc(',', fp);
                    }
                }
                fputc('\n', fp);
            }
            fputc('\n', fp);
        }
    }

    fclose(fp);
    return 0;
}

static void init_kernels(void) {
    memset(kernel, 0, sizeof(kernel));

    kernel[0][0][0][1] = 1.0;  gmap[0][0] = 1;
    kernel[0][1][0][2] = 1.0;  gmap[0][1] = 2;
    kernel[0][2][0][0] = 1.0;  gmap[0][2] = 8;
    kernel[0][3][2][1] = 1.0;  gmap[0][3] = 1;
    kernel[0][4][2][2] = 1.0;  gmap[0][4] = 8;
    kernel[0][5][2][0] = 1.0;  gmap[0][5] = 2;

    kernel[1][0][0][2] = 1.0;  gmap[1][0] = 2;
    kernel[1][1][1][2] = 1.0;  gmap[1][1] = 3;
    kernel[1][2][0][1] = 1.0;  gmap[1][2] = 1;
    kernel[1][3][2][0] = 1.0;  gmap[1][3] = 2;
    kernel[1][4][2][1] = 1.0;  gmap[1][4] = 1;
    kernel[1][5][1][0] = 1.0;  gmap[1][5] = 3;

    kernel[2][0][1][2] = 1.0;  gmap[2][0] = 3;
    kernel[2][1][2][2] = 1.0;  gmap[2][1] = 4;
    kernel[2][2][0][2] = 1.0;  gmap[2][2] = 2;
    kernel[2][3][1][0] = 1.0;  gmap[2][3] = 3;
    kernel[2][4][2][0] = 1.0;  gmap[2][4] = 2;
    kernel[2][5][0][0] = 1.0;  gmap[2][5] = 4;

    kernel[3][0][2][2] = 1.0;  gmap[3][0] = 4;
    kernel[3][1][2][1] = 1.0;  gmap[3][1] = 5;
    kernel[3][2][1][2] = 1.0;  gmap[3][2] = 3;
    kernel[3][3][0][0] = 1.0;  gmap[3][3] = 4;
    kernel[3][4][1][0] = 1.0;  gmap[3][4] = 3;
    kernel[3][5][0][1] = 1.0;  gmap[3][5] = 5;

    kernel[4][0][2][1] = 1.0;  gmap[4][0] = 5;
    kernel[4][1][2][0] = 1.0;  gmap[4][1] = 6;
    kernel[4][2][2][2] = 1.0;  gmap[4][2] = 4;
    kernel[4][3][0][1] = 1.0;  gmap[4][3] = 5;
    kernel[4][4][0][0] = 1.0;  gmap[4][4] = 4;
    kernel[4][5][0][2] = 1.0;  gmap[4][5] = 6;

    kernel[5][0][2][0] = 1.0;  gmap[5][0] = 6;
    kernel[5][1][1][0] = 1.0;  gmap[5][1] = 7;
    kernel[5][2][2][1] = 1.0;  gmap[5][2] = 5;
    kernel[5][3][0][2] = 1.0;  gmap[5][3] = 6;
    kernel[5][4][0][1] = 1.0;  gmap[5][4] = 5;
    kernel[5][5][1][2] = 1.0;  gmap[5][5] = 7;

    kernel[6][0][1][0] = 1.0;  gmap[6][0] = 7;
    kernel[6][1][0][0] = 1.0;  gmap[6][1] = 8;
    kernel[6][2][2][0] = 1.0;  gmap[6][2] = 6;
    kernel[6][3][1][2] = 1.0;  gmap[6][3] = 7;
    kernel[6][4][0][2] = 1.0;  gmap[6][4] = 6;
    kernel[6][5][2][2] = 1.0;  gmap[6][5] = 8;

    kernel[7][0][0][0] = 1.0;  gmap[7][0] = 8;
    kernel[7][1][0][1] = 1.0;  gmap[7][1] = 1;
    kernel[7][2][1][0] = 1.0;  gmap[7][2] = 7;
    kernel[7][3][2][2] = 1.0;  gmap[7][3] = 8;
    kernel[7][4][1][2] = 1.0;  gmap[7][4] = 7;
    kernel[7][5][2][1] = 1.0;  gmap[7][5] = 1;
}

static void configure_transitions(void) {
    for (int j = 0; j < n_orient; ++j) {
        for (int i = 0; i < n_actions; ++i) {
            int tgt = gmap[j][i];
            if (tgt <= 0) {
                gmap[j][i] = 1;
                continue;
            }
            int wrapped = ((tgt - 1) % n_orient) + 1;
            gmap[j][i] = wrapped;
        }
    }
}

static double logsumexp(const double *vals, int n) {
    double maxv = vals[0];
    for (int i = 1; i < n; ++i) {
        if (vals[i] > maxv) maxv = vals[i];
    }
    if (maxv == -INFINITY) {
        return -INFINITY;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += exp(vals[i] - maxv);
    return maxv + log(sum);
}

int main(int argc, char **argv) {
    static double V[MAX_ORIENT][H + 2][W + 2];
    static double V_prev[MAX_ORIENT][H + 2][W + 2];
    static double Q[MAX_ORIENT][MAX_ACTIONS][H][W];
    static double reward[H][W];
    static double policy[MAX_ORIENT][MAX_ACTIONS][H][W];

    const char *reward_path = "reward.csv";
    for (int idx = 1; idx < argc; ++idx) {
        const char *arg = argv[idx];
        if (strcmp(arg, "--reward") == 0) {
            if (idx + 1 >= argc) {
                fprintf(stderr, "--reward expects a file path\n");
                return 1;
            }
            reward_path = argv[++idx];
        } else if (strcmp(arg, "--k-max") == 0) {
            if (idx + 1 >= argc || parse_int_arg(arg, argv[idx + 1], &k_max) != 0) {
                return 1;
            }
            ++idx;
        } else if (strcmp(arg, "--tol") == 0) {
            double tval = tol;
            if (idx + 1 >= argc || parse_double_arg(arg, argv[idx + 1], &tval) != 0) {
                return 1;
            }
            tol = tval;
            ++idx;
        } else if (strcmp(arg, "--n-orient") == 0) {
            if (idx + 1 >= argc || parse_int_arg(arg, argv[idx + 1], &n_orient) != 0) {
                return 1;
            }
            ++idx;
        } else if (strcmp(arg, "--n-actions") == 0) {
            if (idx + 1 >= argc || parse_int_arg(arg, argv[idx + 1], &n_actions) != 0) {
                return 1;
            }
            ++idx;
        } else if (strcmp(arg, "--allowed-actions") == 0) {
            if (idx + 1 >= argc || parse_int_arg(arg, argv[idx + 1], &n_allowed_actions) != 0) {
                return 1;
            }
            ++idx;
        } else if (strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            reward_path = arg;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (k_max <= 0) {
        fprintf(stderr, "k-max must be positive\n");
        return 1;
    }
    if (tol <= 0.0) {
        fprintf(stderr, "tol must be positive\n");
        return 1;
    }
    if (n_orient < 1 || n_orient > MAX_ORIENT) {
        fprintf(stderr, "n-orient must be between 1 and %d\n", MAX_ORIENT);
        return 1;
    }
    if (n_actions < 1 || n_actions > MAX_ACTIONS) {
        fprintf(stderr, "n-actions must be between 1 and %d\n", MAX_ACTIONS);
        return 1;
    }
    if (n_allowed_actions < 1 || n_allowed_actions > n_actions) {
        fprintf(stderr, "allowed-actions must be between 1 and n-actions (%d)\n", n_actions);
        return 1;
    }

    init_kernels();
    configure_transitions();

    for (int j = 0; j < MAX_ORIENT; ++j) {
        for (int y = 0; y < H + 2; ++y) {
            for (int x = 0; x < W + 2; ++x) {
                V[j][y][x] = -1e9;
                V_prev[j][y][x] = -1e9;
            }
        }
    }

    if (load_reward_csv(reward_path, reward) != 0) {
        return 1;
    }

    const int goal_x = W / 2;
    const int goal_y = H / 2;
    int iters_used = k_max;
    int converged = 0;

    for (int t = 0; t < k_max; ++t) {
        for (int j = 0; j < n_orient; ++j) {
            memcpy(V_prev[j], V[j], sizeof(double) * (H + 2) * (W + 2));
            V[j][goal_y + 1][goal_x + 1] = 0.0;
        }

        for (int j = 0; j < n_orient; ++j) {
            for (int i = 0; i < n_actions; ++i) {
                if (!action_allowed(i)) {
                    for (int y = 0; y < H; ++y) {
                        for (int x = 0; x < W; ++x) {
                            Q[j][i][y][x] = -INFINITY;
                        }
                    }
                    continue;
                }
                int gj = gmap[j][i] - 1;
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {
                        double acc = 0.0;
                        for (int ky = 0; ky < 3; ++ky) {
                            for (int kx = 0; kx < 3; ++kx) {
                                double w = kernel[j][i][ky][kx];
                                if (w != 0.0) {
                                    acc += V[gj][y + ky][x + kx] * w;
                                }
                            }
                        }
                        Q[j][i][y][x] = reward[y][x] + acc;
                    }
                }
            }
        }

        for (int j = 0; j < n_orient; ++j) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    double vals[MAX_ACTIONS];
                    for (int i = 0; i < n_actions; ++i) {
                        vals[i] = action_allowed(i) ? Q[j][i][y][x] : -INFINITY;
                    }
                    V[j][y + 1][x + 1] = logsumexp(vals, n_actions);
                }
            }

            for (int y = 0; y < H + 2; ++y) {
                V[j][y][0] = -INFINITY;
                V[j][y][W + 1] = -INFINITY;
            }
            for (int x = 0; x < W + 2; ++x) {
                V[j][0][x] = -INFINITY;
                V[j][H + 1][x] = -INFINITY;
            }
        }

        double max_delta = 0.0;
        for (int j = 0; j < n_orient; ++j) {
            for (int y = 0; y < H + 2; ++y) {
                for (int x = 0; x < W + 2; ++x) {
                    double delta = fabs(V[j][y][x] - V_prev[j][y][x]);
                    if (delta > max_delta) {
                        max_delta = delta;
                    }
                }
            }
        }

        if (max_delta < tol) {
            converged = 1;
            iters_used = t + 1;
            fprintf(stderr, "Converged at iteration %d (delta=%.6g)\n", iters_used, max_delta);
            break;
        }
        
        if ((t + 1) % 100 == 0) {
            fprintf(stderr, "[t=%d] max_delta=%.6g\n", t + 1, max_delta);
        }
    }

    if (!converged) {
        fprintf(stderr, "Warning: value iteration hit k-max=%d without reaching tol=%.4g\n", k_max, tol);
    }

    for (int j = 0; j < n_orient; ++j) {
        for (int i = 0; i < n_actions; ++i) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    if (!action_allowed(i)) {
                        policy[j][i][y][x] = 0.0;
                    } else {
                        policy[j][i][y][x] = exp(Q[j][i][y][x] - V[j][y + 1][x + 1]);
                    }
                }
            }
        }
    }

    if (write_policy_txt("policy.txt", policy, iters_used) != 0) {
        return 1;
    }

    // Also save policy to map folder
    char *map_folder = extract_map_folder(reward_path);
    char policy_in_map[512];
    snprintf(policy_in_map, sizeof(policy_in_map), "%s/policy.txt", map_folder);
    if (write_policy_txt(policy_in_map, policy, iters_used) != 0) {
        fprintf(stderr, "Warning: could not save policy to map folder: %s\n", policy_in_map);
    }

    printf("RLConvNet simple demo. V(goal)=%.6f iters=%d %s tol=%.4g\n",
           V[0][goal_y + 1][goal_x + 1], iters_used, converged ? "(converged)" : "(maxed)", tol);
    printf("Policy saved to: policy.txt and %s\n", policy_in_map);

    return 0;
}
