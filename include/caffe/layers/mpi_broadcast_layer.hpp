
#ifndef CAFFE_MPI_BROADCAST_LAYER_HPP_
#define CAFFE_MPI_BROADCAST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/mpi_base_layer.hpp"


namespace caffe{

template <typename Dtype>
class MPIBroadcastLayer : public MPIBaseLayer<Dtype> {
 public:
  explicit MPIBroadcastLayer(const LayerParameter& param)
    : MPIBaseLayer<Dtype>(param){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "MPIBroadcast"; }
	bool MPISyncFlag(bool flag);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}

#endif