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
CPPFLAGS  += -I../api -I../mat -I../src/mat -I../src/nn -I../src/lp -I../src/image -I../src/math -I../import/image_io
LDLIBS    ?= -lm
STD       := $(shell printf '' | $(CC) -std=c23 -E -x c - >/dev/null 2>&1 && echo c23 || echo c2x)

UNAME_M   := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  ARCH_CFLAGS := -mavx2 -mfma
endif

OBJDIR    := ../obj
BINDIR    := .

CORE      := fiv_common fiv_ctensor fiv_binary_op
MAT       := fiv_mat_transpose fiv_mat_vec fiv_mat_vec_db fiv_mat_mul fiv_mat_mul_db fiv_mat_reduce_db fiv_linalg_kernels fiv_mat_cholesky fiv_mat_lu fiv_mat_eig_sym fiv_mat_svd
NN        := fiv_nn_infer fiv_nn_topo fiv_nn_train fiv_linear_node fiv_activate_fn fiv_nn_conv2d fiv_nn_1x1_conv2d fiv_nn_2x2_conv2d fiv_nn_3x3_conv2d fiv_nn_5x5_conv2d fiv_flatten_node fiv_max_2d fiv_add_node fiv_pad_node fiv_concat_node fiv_prelu_node fiv_sigmoid_node fiv_spatial_pad_node fiv_upsample_node
CONV2D_KERNEL_OBJS := $(addprefix $(OBJDIR)/,fiv_nn_1x1_conv2d.o fiv_nn_2x2_conv2d.o fiv_nn_3x3_conv2d.o fiv_nn_5x5_conv2d.o)
NEW_NN_OBJS := $(addprefix $(OBJDIR)/,fiv_concat_node.o fiv_prelu_node.o fiv_sigmoid_node.o fiv_spatial_pad_node.o fiv_upsample_node.o)
MATH      := fiv_math fiv_math_sigmoid fiv_math_softmax fiv_math_swiglu
MATH_OBJS := $(addprefix $(OBJDIR)/,fiv_math.o fiv_math_sigmoid.o fiv_math_softmax.o fiv_math_swiglu.o)
TESTS     := test_darray test_ctensor test_mat_transpose test_mat_vec test_mat_mul_vec_db test_mat_mul test_mat_mul_db test_mat_reduce_db test_mat_cholesky test_mat_lu test_mat_eig_sym test_mat_svd test_mat_svd_jacobi test_nn test_nn_mnist test_nn_conv2d test_nn_mnist_conv test_blazeface test_image_io test_color_space_perf test_data_convert test_face_api test_landmark_ops test_landmark_net test_landmark_api test_landmark_e2e test_yunet_api test_yunet_resize test_lp_sp_matrix test_lp_ops test_lp_rescale test_lp_pdhg test_lp_solve test_lp_solve13

APPFACE   := ../app/face/blazeFace
LANDMARK  := ../app/face/landmark
YUNET     := ../app/face/yunet
APPINC    := -I$(APPFACE) -I$(LANDMARK) -I$(YUNET)
APPFACE_OBJS := $(OBJDIR)/fiv_face_weights.o $(OBJDIR)/fiv_face_warp.o $(OBJDIR)/fiv_face.o
LANDMARK_OBJS := $(addprefix $(OBJDIR)/,fiv_landmark.o fiv_landmark_geom.o fiv_landmark_model.o)
YUNET_OBJS    := $(addprefix $(OBJDIR)/,fiv_yunet.o fiv_yunet_model.o)

# object sets shared by the face/landmark/yunet app tests
NN_ENGINE  := $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(CONV2D_KERNEL_OBJS) $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(NEW_NN_OBJS)
IMAGE_OBJS := $(OBJDIR)/fiv_image_io.o $(OBJDIR)/fiv_image_color_space.o $(OBJDIR)/fiv_image_resizer.o
BASE_OBJS  := $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

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
test_mat_cholesky: $(OBJDIR)/test_mat_cholesky.o $(OBJDIR)/fiv_linalg_kernels.o $(OBJDIR)/fiv_mat_cholesky.o $(OBJDIR)/fiv_mat_lu.o $(OBJDIR)/fiv_mat_transpose.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_lu: $(OBJDIR)/test_mat_lu.o $(OBJDIR)/fiv_linalg_kernels.o $(OBJDIR)/fiv_mat_cholesky.o $(OBJDIR)/fiv_mat_lu.o $(OBJDIR)/fiv_mat_transpose.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_eig_sym: $(OBJDIR)/test_mat_eig_sym.o $(OBJDIR)/fiv_linalg_kernels.o $(OBJDIR)/fiv_mat_eig_sym.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_svd: $(OBJDIR)/test_mat_svd.o $(OBJDIR)/fiv_mat_svd.o $(OBJDIR)/fiv_mat_transpose.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_mat_svd_jacobi: $(OBJDIR)/test_mat_svd_jacobi.o $(OBJDIR)/fiv_mat_svd.o $(OBJDIR)/fiv_mat_transpose.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn:            $(OBJDIR)/test_nn.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(CONV2D_KERNEL_OBJS) $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(NEW_NN_OBJS) $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn_mnist:      $(OBJDIR)/test_nn_mnist.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(CONV2D_KERNEL_OBJS) $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(NEW_NN_OBJS) $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn_conv2d:     $(OBJDIR)/test_nn_conv2d.o $(OBJDIR)/fiv_nn_conv2d.o $(CONV2D_KERNEL_OBJS) $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_nn_mnist_conv: $(OBJDIR)/test_nn_mnist_conv.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(CONV2D_KERNEL_OBJS) $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(NEW_NN_OBJS) $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_blazeface:     $(OBJDIR)/test_blazeface.o $(REFOBS) $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(CONV2D_KERNEL_OBJS) $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(NEW_NN_OBJS) $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_image_io:      $(OBJDIR)/test_image_io.o $(OBJDIR)/fiv_image_io.o $(OBJDIR)/fiv_image_color_space.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_color_space_perf: $(OBJDIR)/test_color_space_perf.o $(OBJDIR)/fiv_image_color_space.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
test_data_convert: $(OBJDIR)/test_data_convert.o $(OBJDIR)/fiv_data_convert.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

test_face_api: $(OBJDIR)/test_face_api.o $(APPFACE_OBJS) $(OBJDIR)/fiv_image_io.o $(OBJDIR)/fiv_image_color_space.o $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(CONV2D_KERNEL_OBJS) $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(NEW_NN_OBJS) $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_mat_mul.o $(OBJDIR)/fiv_mat_mul_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

test_landmark_ops: $(OBJDIR)/test_landmark_ops.o $(OBJDIR)/fiv_prelu_node.o $(OBJDIR)/fiv_concat_node.o $(OBJDIR)/fiv_spatial_pad_node.o $(OBJDIR)/fiv_nn_conv2d.o $(CONV2D_KERNEL_OBJS) $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(NEW_NN_OBJS) $(BASE_OBJS)

test_landmark_net: $(OBJDIR)/test_landmark_net.o $(LANDMARK_OBJS) $(APPFACE_OBJS) $(NN_ENGINE) $(BASE_OBJS)

test_landmark_api: $(OBJDIR)/test_landmark_api.o $(LANDMARK_OBJS) $(APPFACE_OBJS) $(NN_ENGINE) $(IMAGE_OBJS) $(BASE_OBJS)

test_landmark_e2e: $(OBJDIR)/test_landmark_e2e.o $(LANDMARK_OBJS) $(APPFACE_OBJS) $(NN_ENGINE) $(IMAGE_OBJS) $(BASE_OBJS)

test_yunet_api: $(OBJDIR)/test_yunet_api.o $(YUNET_OBJS) $(NN_ENGINE) $(IMAGE_OBJS) $(BASE_OBJS)

test_yunet_resize: $(OBJDIR)/test_yunet_resize.o $(YUNET_OBJS) $(NN_ENGINE) $(IMAGE_OBJS) $(BASE_OBJS)

test_lp_sp_matrix: $(OBJDIR)/test_lp_sp_matrix.o $(OBJDIR)/fiv_sp_matrix.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

test_lp_ops: $(OBJDIR)/test_lp_ops.o $(OBJDIR)/fiv_lp_vec.o $(OBJDIR)/fiv_lp_mat.o $(OBJDIR)/fiv_sp_matrix.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

test_lp_rescale: $(OBJDIR)/test_lp_rescale.o $(OBJDIR)/fiv_lp_rescale.o $(OBJDIR)/fiv_lp_mat.o $(OBJDIR)/fiv_lp_vec.o $(OBJDIR)/fiv_sp_matrix.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

test_lp_pdhg: $(OBJDIR)/test_lp_pdhg.o $(OBJDIR)/fiv_lp_pdhg.o $(OBJDIR)/fiv_lp_mat.o $(OBJDIR)/fiv_lp_vec.o $(OBJDIR)/fiv_sp_matrix.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

test_lp_solve: $(OBJDIR)/test_lp_solve.o $(OBJDIR)/fiv_lp_solve.o $(OBJDIR)/fiv_lp_pdhg.o $(OBJDIR)/fiv_lp_rescale.o $(OBJDIR)/fiv_lp_mat.o $(OBJDIR)/fiv_lp_vec.o $(OBJDIR)/fiv_sp_matrix.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o

test_lp_solve13: $(OBJDIR)/test_lp_solve13.o $(OBJDIR)/fiv_lp_solve.o $(OBJDIR)/fiv_lp_pdhg.o $(OBJDIR)/fiv_lp_rescale.o $(OBJDIR)/fiv_lp_mat.o $(OBJDIR)/fiv_lp_vec.o $(OBJDIR)/fiv_sp_matrix.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o


bench_vec_dot: $(OBJDIR)/bench_vec_dot.o $(OBJDIR)/fiv_mat_vec.o $(OBJDIR)/fiv_mat_vec_db.o $(OBJDIR)/fiv_mat_reduce_db.o $(OBJDIR)/fiv_ctensor.o $(OBJDIR)/fiv_binary_op.o $(OBJDIR)/fiv_common.o
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) -o $(BINDIR)/$@ $^ $(LDLIBS)

# app/face sources live in three sibling dirs; vpath resolves %.c across all of them
vpath %.c $(APPFACE) $(LANDMARK) $(YUNET)
$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) $(APPINC) -c $< -o $@

$(OBJDIR)/test_face_api.o: ../src/test/test_face_api.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) $(APPINC) -c $< -o $@

$(OBJDIR)/test_landmark_api.o: ../src/test/test_landmark_api.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) $(APPINC) -c $< -o $@

$(OBJDIR)/test_landmark_net.o: ../src/test/test_landmark_net.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) $(APPINC) -c $< -o $@

$(OBJDIR)/test_landmark_e2e.o: ../src/test/test_landmark_e2e.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) $(APPINC) -c $< -o $@

$(OBJDIR)/test_yunet_api.o: ../src/test/test_yunet_api.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) $(APPINC) -c $< -o $@

$(OBJDIR)/test_yunet_resize.o: ../src/test/test_yunet_resize.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) $(APPINC) -c $< -o $@

.PHONY: all run clean bench_vec_dot oracle oracle13
all: $(TESTS) $(OBJDIR)/fiv_nn_infer.o $(OBJDIR)/fiv_nn_topo.o $(OBJDIR)/fiv_nn_train.o $(OBJDIR)/fiv_linear_node.o $(OBJDIR)/fiv_activate_fn.o $(OBJDIR)/fiv_nn_conv2d.o $(CONV2D_KERNEL_OBJS) $(OBJDIR)/fiv_flatten_node.o $(OBJDIR)/fiv_max_2d.o $(OBJDIR)/fiv_add_node.o $(OBJDIR)/fiv_pad_node.o $(NEW_NN_OBJS) $(MATH_OBJS)

$(OBJDIR)/%.o: ../src/ctensor/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/mat/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/nn/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/math/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/image/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/lp/extra_test/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/%.o: ../src/lp/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

# Explicit rules for lp objects (avoid pattern-rule collision with src/test).
$(OBJDIR)/fiv_sp_matrix.o: ../src/lp/fiv_sp_matrix.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/test_lp_sp_matrix.o: ../src/lp/extra_test/test_sp_matrix.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/fiv_lp_vec.o: ../src/lp/fiv_lp_vec.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/fiv_lp_mat.o: ../src/lp/fiv_lp_mat.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/test_lp_ops.o: ../src/lp/extra_test/test_lp_ops.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/fiv_lp_rescale.o: ../src/lp/fiv_lp_rescale.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/test_lp_rescale.o: ../src/lp/extra_test/test_lp_rescale.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/fiv_lp_pdhg.o: ../src/lp/fiv_lp_pdhg.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -std=$(STD) $(CPPFLAGS) -c $< -o $@

$(OBJDIR)/test_lp_pdhg.o: ../src/lp/extra_test/test_lp_pdhg.c | $(OBJDIR)
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

# Cross-language oracle: C driver vs the real torch pdlp.py on the same problems.
oracle: test_lp_solve
	./test_lp_solve > /tmp/c_oracle.txt
	python3 ../src/lp/extra_test/oracle_lp_solve.py > /tmp/py_oracle.txt
	python3 ../src/lp/extra_test/compare_oracle.py /tmp/c_oracle.txt /tmp/py_oracle.txt

# M5 oracle: the 13 standard LP problems, C port (dense+sparse) vs torch pdlp.py.
oracle13: test_lp_solve13
	./test_lp_solve13 > /tmp/c_oracle13.txt
	python3 ../src/lp/extra_test/oracle_lp_solve13.py > /tmp/py_oracle13.txt
	python3 ../src/lp/extra_test/compare_oracle13.py /tmp/c_oracle13.txt /tmp/py_oracle13.txt

clean:
	rm -f $(OBJDIR)/*.o $(addprefix $(BINDIR)/,$(TESTS))
