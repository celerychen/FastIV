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
CPPFLAGS  += -I../api -I../mat -I../src/mat -I../src/nn -I../src/image -I../import/image_io
LDLIBS    ?= -lm
STD       := $(shell printf '' | $(CC) -std=c23 -E -x c - >/dev/null 2>&1 && echo c23 || echo c2x)

UNAME_M   := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  ARCH_CFLAGS := -mavx2 -mfma
endif

OBJDIR    := ../obj
BINDIR    := .

CORE      := fiv_common fiv_ctensor fiv_binary_op
MAT       := fiv_mat_transpose fiv_mat_vec fiv_mat_vec_db fiv_mat_mul fiv_mat_mul_db fiv_mat_reduce_db
NN        := fiv_nn_infer fiv_nn_topo fiv_nn_train fiv_linear_node fiv_activate_fn fiv_nn_conv2d fiv_flatten_node fiv_max_2d fiv_add_node fiv_pad_node
TESTS     := test_darray test_ctensor test_mat_transpose test_mat_vec test_mat_mul_vec_db test_mat_mul test_mat_mul_db test_mat_reduce_db test_nn test_nn_mnist test_nn_conv2d test_nn_mnist_conv test_blazeface test_image_io test_color_space_perf test_data_convert test_face_api

APPFACE   := ../app/face
APPFACE_OBJS := $(OBJDIR)/fiv_face_weights.o $(OBJDIR)/fiv_face_warp.o $(OBJDIR)/fiv_face.o

REFDIR    := ../src/reference/c_face_detect_release/c
REFOBS    := $(OBJDIR)/ref_weights.o $(OBJDIR)/ref_geom.o $(OBJDIR)/ref_cnn_ops.o $(OBJDIR)/ref_detect.o

.DEFAULT_GOAL := all

OBJ       = $(OBJDIR)/$*.o

test_darray:        $(OBJDIR)/test_darray.o $(OBJDIR)/fiv_common.o
test_ctensor:       $(OBJDIR)/test_ctensor.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_transpose: $(OBJDIR)/test_mat_transpose.o $(OBJDIR)/fiv_mat_transpose.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_vec:       $(OBJDIR)/test_mat_vec.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_mul_vec_db: $(OBJDIR)/test_mat_mul_vec_db.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_mul:       $(OBJDIR)/test_mat_mul.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_mul_db:     $(OBJDIR)/test_mat_mul_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_reduce_db: $(OBJDIR)/test_mat_reduce_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn:            $(OBJDIR)/test_nn.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn_mnist:      $(OBJDIR)/test_nn_mnist.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn_conv2d:     $(OBJDIR)/test_nn_conv2d.o $(OBJDIR)/fiv_nn_conv2d.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn_mnist_conv: $(OBJDIR)/test_nn_mnist_conv.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_blazeface:     $(OBJDIR)/test_blazeface.o $(REFOBS) $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_image_io:      $(OBJDIR)/test_image_io.o $(OBJDIR)/fiv_image_io.o $(OBJDIR)/fiv_image_color_space.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_color_space_perf: $(OBJDIR)/test_color_space_perf.o $(OBJDIR)/fiv_image_color_space.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_data_convert: $(OBJDIR)/test_data_convert.o $(OBJDIR)/fiv_data_convert.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

test_face_api: $(OBJDIR)/test_face_api.o $(APPFACE_OBJS) $(OBJDIR)/fiv_image_io.o $(OBJDIR)/fiv_image_color_space.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

$(OBJDIR)/fiv_face.o: $(APPFACE)/fiv_face.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -I$(APPFACE) -c $< -o $@

$(OBJDIR)/fiv_face_weights.o: $(APPFACE)/fiv_face_weights.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -I$(APPFACE) -c $< -o $@

$(OBJDIR)/fiv_face_warp.o: $(APPFACE)/fiv_face_warp.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -I$(APPFACE) -c $< -o $@

$(OBJDIR)/test_face_api.o: ../src/test/test_face_api.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -I$(APPFACE) -c $< -o $@

.PHONY: all run clean
all: $(TESTS) $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o

$(OBJDIR)/%.o: ../src/ctensor/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/mat/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/nn/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/image/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/test/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/ref_%.o: $(REFDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) -I$(REFDIR) -c $< -o $@

$(OBJDIR)/test_blazeface.o: ../src/test/test_blazeface.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -I$(REFDIR) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TESTS): %: $(OBJDIR)/%.o
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) -o $(BINDIR)/$@ $^ $(LDLIBS)

run: all
	@for t in $(TESTS); do echo "=== $$t ==="; ./$$t; done

clean:
	rm -f $(OBJDIR)/*.o $(addprefix $(BINDIR)/,$(TESTS))
