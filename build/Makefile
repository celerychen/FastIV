# build/Makefile - build all FastIV correctness tests (macOS / Linux)
#   make            -> build all test binaries
#   make run        -> build and run all tests
#   make clean      -> remove intermediates and test binaries
#
# Layout: api/*.h (public headers), src/ctensor + src/mat (implementations),
#         src/test/*.c (tests), obj/ (intermediates), build/ (binaries).
#
# Platform notes:
#   - C23 typed enums need -std=c23; clang < 17 spells it -std=c2x, so the
#     supported spelling is detected automatically.
#   - On x86-64 hosts AVX2+FMA is enabled via -mavx2 -mfma; on ARM hosts the
#     NEON kernels are selected automatically by __aarch64__.

CC        ?= cc
CFLAGS    ?= -O2 -Wall
CPPFLAGS  += -I../api -I../mat
LDLIBS    ?= -lm
STD       := $(shell printf '' | $(CC) -std=c23 -E -x c - >/dev/null 2>&1 && echo c23 || echo c2x)

UNAME_M   := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  ARCH_CFLAGS := -mavx2 -mfma
endif

OBJDIR    := ../obj
BINDIR    := .

CORE      := fiv_common fiv_ctensor fiv_binary_op
MAT       := fiv_mat_transpose fiv_mat_vec fiv_mat_mul
NN        := fiv_nn_infer fiv_nn_topo fiv_nn_train fiv_linear_node fiv_activate_fn
TESTS     := test_darray test_ctensor test_mat_transpose test_mat_vec test_mat_mul test_nn test_nn_mnist

.DEFAULT_GOAL := all

OBJ       = $(OBJDIR)/$*.o

test_darray:        $(OBJDIR)/test_darray.o $(OBJDIR)/fiv_common.o
test_ctensor:       $(OBJDIR)/test_ctensor.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_transpose: $(OBJDIR)/test_mat_transpose.o $(OBJDIR)/fiv_mat_transpose.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_vec:       $(OBJDIR)/test_mat_vec.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_mul:       $(OBJDIR)/test_mat_mul.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn:            $(OBJDIR)/test_nn.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn_mnist:      $(OBJDIR)/test_nn_mnist.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

.PHONY: all run clean
all: $(TESTS) $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o

$(OBJDIR)/%.o: ../src/ctensor/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/mat/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/nn/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/test/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TESTS): %: $(OBJDIR)/%.o
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) -o $(BINDIR)/$@ $^ $(LDLIBS)

run: all
	@for t in $(TESTS); do echo "=== $$t ==="; ./$$t; done

clean:
	rm -f $(OBJDIR)/*.o $(addprefix $(BINDIR)/,$(TESTS))
