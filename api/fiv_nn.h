/*
 * FastIV - Fast image and vision
 * Copyright (C) 2026 Celery Chen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * See LICENSE file in project root for full license text.
 */

#ifndef _FIV_NN_H_
#define _FIV_NN_H_

#include "fiv_data_typedefs.h"
#include "fiv_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    FIV_NN_NODE_INPUT = 0,
    FIV_NN_NODE_LINEAR,
    FIV_NN_NODE_RELU,
    FIV_NN_NODE_RELU6,
    FIV_NN_NODE_CONV2D_STD,
    FIV_NN_NODE_CONV2D_DEPTHWISE,
    FIV_NN_NODE_CONV2D_POINTWISE,
    FIV_NN_NODE_CONV2D_SEPARABLE,
    FIV_NN_NODE_FLATTEN,
    FIV_NN_NODE_MAX2D,
    FIV_NN_NODE_ADD,
    FIV_NN_NODE_PAD,
    FIV_NN_NODE_UPSAMPLE2X,   /* nearest-neighbor x2 upsample (FPN lateral) */
    FIV_NN_NODE_SIGMOID,      /* element-wise sigmoid activation */
    FIV_NN_NODE_PRELU,        /* parametric ReLU (per-channel learned slope) */
    FIV_NN_NODE_CONCAT,       /* concatenate multiple tensors along `axis` (multi-input) */
    FIV_NN_NODE_SPATIAL_PAD,  /* constant(spatial) zero-pad HxW with explicit margins */
    FIV_NN_NODE_TYPE_NUM
} fiv_nn_node_type;


typedef struct {
    int in_features;   /* input feature count */
    int out_features;  /* output feature count */
} fiv_linear_node_params;


typedef struct{
   int conv2d_method; /* CONV2D_STD,CONV2D_DEPTHWISE,CONV2D_POINTWISE,CONV2D_SEPARABLE; */
   int kernel_size_x;
   int kernel_size_y;
   int stride;
   int padding_method;  /* 0 = zero fill, 1 = replicate edge element */
   int input_channels;
   int output_channels;
   int bias;  /* 0 = no bias, 1 = per-output-channel bias */
   int pad_top;    /* explicit start padding (rows) */
   int pad_bottom; /* explicit end padding (rows) */
   int pad_left;   /* explicit start padding (cols) */
   int pad_right;  /* explicit end padding (cols) */
}fiv_conv2d_params;

/* Channel-pad params: NCHW, appends zero channels at the END of the channel
   dim so the output has exactly output_channels (output_channels >= in_c). */
typedef struct {
    int output_channels;
} fiv_pad_node_params;

/* PReLU (parametric ReLU) params: applies alpha[c] * x for x < 0 per channel,
   identity for x >= 0. channels must equal the tensor's channel count. alpha is
   an optional per-channel slope array (copied by the op); NULL initialises to
   0.25. Used as the FaceMesh landmark backbone activation. */
typedef struct {
    int          channels;  /* number of per-channel slopes = input channels */
    const ivf32* alpha;     /* optional initial slopes; NULL -> init 0.25f */
} fiv_prelu_node_params;

/* CONCAT params: concatenates the (NCHW) input tensors along `axis`. For the
   FaceMesh landmark net the concat axis is the channel dim (NCHW axis=1), but
   any valid axis is supported. output_channels is the accumulated channel count
   of all inputs (used to size the output and validated per forward). */
typedef struct {
    int axis;             /* concat axis, NCHW (default 1 = channels) */
    int output_channels;  /* sum of all inputs' channel counts */
} fiv_concat_node_params;

/* Spatial-pad params: NCHW, inserts `value` (default 0) along the H/W borders
   by the symmetric amounts pad_top/pad_bottom/pad_left/pad_right, extending the
   tensor spatially (unlike fiv_pad_node_params which pads channels). */
typedef struct {
    int   pad_top;
    int   pad_bottom;
    int   pad_left;
    int   pad_right;
    ivf32 value;  /* fill value for padded cells (default 0) */
} fiv_spatial_pad_node_params;









/* Create an empty network context; returns NULL on allocation failure. */
void* fiv_create_neural_network();

/* Add a node to the network in order.
   nn_context  - network context from fiv_create_neural_network
   node_type   - FIV_NN_NODE_* (INPUT / LINEAR / RELU / RELU6)
   index_start - source node id feeding this node (0 = external input)
   index_end   - this node's own id (must equal the current node count)
   params      - op-specific params (e.g. fiv_linear_node_params); NULL for INPUT/RELU */
fiv_ret fiv_neural_network_add_node(void* nn_context, int node_type, int index_start, int index_end, void* params);

/* Add a node fed by MULTIPLE source nodes (e.g. ADD residual: main path +
   shortcut). index_starts lists every source node id in order; index_starts[0]
   is the primary source and is what single-input nodes would use. All other
   add_node call sites are unchanged; only multi-input nodes (ADD) need this. */
fiv_ret fiv_neural_network_add_node_multi(void* nn_context, int node_type,
                                          const int* index_starts, int num_src,
                                          int index_end, void* params);

/* Run inference for one input and write the result into output.
   output     - result tensor (must match the network output shape)
   input      - input tensor fed to node 0
   nn_context - network context (built or loaded from a model) */
fiv_ret fiv_neural_network_inference(void* output, void* input, void* nn_context);

/* Release a network context and free all internal state.
   nn_context - address of the context pointer; set to NULL on return */
fiv_ret fiv_release_neural_network(void** nn_context);

/* Serialize the network (graph + weights/biases) to a file.
   model_name - output file path
   nn_context - network context */
fiv_ret fiv_neural_network_save_model(char* model_name, void* nn_context);

/* Load a network previously saved with fiv_neural_network_save_model.
   model_name - input file path; returns NULL on error (missing / bad magic / bad type) */
void* fiv_create_neural_network_from_model(char* model_name);

typedef struct {
   int epoch_num;          /* number of training epochs */
   int bach_size;          /* samples per batch (caller spelling) */
   ivf32 learning_rate;    /* SGD step size */
   int loss_fn_type;       /* 0 = cross-entropy (classification), 1 = MSE (regression) */
}fiv_nn_train_params;






/* Descriptor the training framework passes to the data-loading callback and
   which the callback fills. It carries both the run metadata (name / sizes /
   current epoch) and the current batch's tensors: current_bach_inputs is fed to
   network node 0 and current_batch_outputs is consumed by the loss. Both are
   engine-generic tensors (any dim, 32F); the framework reads their shape by
   fiv_data_id and never assumes a matrix layout or contiguity. */
typedef struct {
   char* data_set_name;   /* caller selects a named set; NULL => default */
   size_t total_data_size;   /* total samples in the set (informational) */
   int feature_size;      /* input feature count (informational) */
   int label_size;        /* label size per sample (informational) */
   int current_epoch_index;       /* current epoch (0-based); filled by the framework */
   int current_batch_index;
   void* current_batch_inputs;   /* this batch's input tensor */
   void* current_batch_outputs; /* this batch's target / label tensor */
}fiv_dadaset_info;






/* User-supplied data loader: each call yields ONE batch. The callback fills
   data_set_info->current_batch_inputs / current_batch_outputs in place and
   returns FIV_RET_OK, or FIV_RET_ERR_END_OF_FILE when the current epoch is
   exhausted (framework starts the next epoch), or FIV_RET_DATA_NOT_FOUND when
   the named set is unavailable. */
typedef fiv_ret (*fiv_load_dataset_fn)(fiv_dadaset_info* data_set_info);



/* Train the network with SGD + backprop.
   nn_context       - network context
   nn_train_params  - epochs / batch size / learning rate / loss type
   load_dataset     - callback that yields one batch per call */
fiv_ret fiv_neural_network_train(void* nn_context, fiv_nn_train_params* nn_train_params, fiv_load_dataset_fn load_dataset);





#ifdef __cplusplus
}
#endif

#endif  /* _FIV_NN_H_ */
