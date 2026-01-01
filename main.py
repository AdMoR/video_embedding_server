# @title Prepare environment

import os
import sys


import jax
from jax.extend import backend
import tensorflow as tf

# Do not let TF use the GPU or TPUs.
#tf.config.set_visible_devices([], "GPU")
#tf.config.set_visible_devices([], "TPU")
print(tf.config.list_physical_devices())

print(f"JAX version:  {jax.__version__}")
print(f"JAX platform: {backend.get_backend().platform}")
print(f"JAX devices:  {jax.device_count()}")
print(jax.devices())

import mediapy
import numpy as np
from PIL import Image


def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
):
  """Reads and preprocesses a video."""

  frames = mediapy.read_video(filename)

  # Sample to target number of frames.
  frame_indices = np.linspace(
      0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
  )
  frames = np.array([frames[i] for i in frame_indices])

  # Resize to target size.
  original_height, original_width = frames.shape[-3:-1]
  target_height, target_width = target_frame_size
  assert (
      original_height * target_width == original_width * target_height
  ), 'Currently does not support aspect ratio mismatch.'
  frames = mediapy.resize_video(frames, shape=target_frame_size)

  # Normalize pixel values to [0.0, 1.0].
  frames = mediapy.to_float01(frames)

  return frames


def compute_similarity_matrix(
    video_embeddings,
    text_embeddings,
    temperature: float,
    apply_softmax: str | None = None,
) -> np.ndarray:
  """Computes cosine similarity matrix."""
  assert apply_softmax in [None, 'over_texts', 'over_videos']
  emb_dim = video_embeddings[0].shape[-1]
  assert emb_dim == text_embeddings[0].shape[-1]

  video_embeddings = np.array(video_embeddings).reshape(-1, emb_dim)
  text_embeddings = np.array(text_embeddings).reshape(-1, emb_dim)
  similarity_matrix = np.dot(video_embeddings, text_embeddings.T)

  if temperature is not None:
    similarity_matrix /= temperature

  if apply_softmax == 'over_videos':
    similarity_matrix = np.exp(similarity_matrix)
    similarity_matrix = similarity_matrix / np.sum(
        similarity_matrix, axis=0, keepdims=True
    )
  elif apply_softmax == 'over_texts':
    similarity_matrix = np.exp(similarity_matrix)
    similarity_matrix = similarity_matrix / np.sum(
        similarity_matrix, axis=1, keepdims=True
    )

  return similarity_matrix


import jax
import jax.numpy as jnp
from videoprism import models as vp

MODEL_NAME = 'videoprism_lvt_public_v1_base'  # @param ['videoprism_lvt_public_v1_base', 'videoprism_lvt_public_v1_large'] {allow-input: false}
USE_BFLOAT16 = False  # @param { type: "boolean" }
NUM_FRAMES = 16
FRAME_SIZE = 288

fprop_dtype = jnp.bfloat16 if USE_BFLOAT16 else None
flax_model = vp.get_model(MODEL_NAME, fprop_dtype=fprop_dtype)
loaded_state = vp.load_pretrained_weights(MODEL_NAME)
text_tokenizer = vp.load_text_tokenizer('c4_en')


@jax.jit
def forward_fn(inputs, text_token_ids, text_paddings, train=False):
  return flax_model.apply(
      loaded_state,
      inputs,
      text_token_ids,
      text_paddings,
      train=train,
  )


# @title Specify input video
VIDEO_FILE_PATH = './videoprism/assets/water_bottle_drumming.mp4'  # @param {type: "string"}

frames = read_and_preprocess_video(
    VIDEO_FILE_PATH,
    target_num_frames=NUM_FRAMES,
    target_frame_size=[FRAME_SIZE, FRAME_SIZE],
)
frames = jnp.asarray(frames[None, ...])  # Add batch dimension.
if USE_BFLOAT16:
  frames = frames.astype(jnp.bfloat16)


# @title Specify input text queries
TEXT_QUERY_CSV = 'playing drums,sitting,playing flute,playing at playground,concert'  # @param {type: "string"}
PROMPT_TEMPLATE = 'a video of {}.'

text_queries = TEXT_QUERY_CSV.split(',')
text_queries = [PROMPT_TEMPLATE.format(t) for t in text_queries]
text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, text_queries)
if USE_BFLOAT16:
  text_paddings = text_paddings.astype(jnp.bfloat16)

print('Input text queries:')
for i, text in enumerate(text_queries):
  print(f'({i + 1}) {text}')

# @title Compute video-to-text retrieval results
import time
tt = time.time()
video_embeddings, text_embeddings, _ = forward_fn(
    frames, text_ids, text_paddings)
print(time.time() - tt)

TEMPERATURE = 0.01  # @param {type: "number"}
similarity_matrix = compute_similarity_matrix(
    video_embeddings,
    text_embeddings,
    temperature=TEMPERATURE,
    apply_softmax='over_texts',
)

v2t_similarity_vector = similarity_matrix[0]
top_indices = np.argsort(v2t_similarity_vector)[::-1]

print(f'Query video: {os.path.basename(VIDEO_FILE_PATH)}')
mediapy.show_video(frames[0].astype(jnp.float32), fps=6.0)

for k, j in enumerate(top_indices):
  print(
      'Top-%d retrieved text: %s [Similarity = %0.4f]'
      % (k + 1, text_queries[j], v2t_similarity_vector[j])
  )
print(f'\nThis is {text_queries[top_indices[0]]}')