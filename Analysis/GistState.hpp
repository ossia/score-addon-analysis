#pragma once
#include <Audio/Settings/Model.hpp>
#include <Gist.h>
#include <ossia/dataflow/audio_port.hpp>
#include <ossia/dataflow/graph_node.hpp>
#include <ossia/dataflow/token_request.hpp>
#include <ossia/dataflow/value_port.hpp>
#include <ossia/network/value/value.hpp>
#include <ossia/detail/flat_map.hpp>
#include <mutex>

namespace ossia::safe_nodes
{
template <typename T>
using timed_vec = ossia::flat_map<int64_t, T>;
}

namespace Analysis
{
struct GistState
{
  // For efficiency we take a reference to the vector<value> member
  // of the ossia variant
  explicit GistState(int bufferSize, int rate):
    out_val{std::vector<ossia::value>{}}
  , output{out_val.v.m_impl.m_value8}
  , bufferSize{bufferSize}
  , rate{rate}
  {
    out_val = output;
    gist.reserve(2);
    gist.emplace_back(bufferSize, rate);
    gist.emplace_back(bufferSize, rate);
  }

  explicit GistState(Audio::Settings::Model& settings):
    GistState{settings.getBufferSize(), settings.getRate()}
  {
  }

  explicit GistState():
    GistState{score::AppContext().settings<Audio::Settings::Model>()}
  {
  }

  ~GistState()
  {
    gist.clear();
  }

  void preprocess(const ossia::audio_port& audio)
  {
    output.resize(audio.samples.size());
    if(gist.size() < audio.samples.size())
    {
      gist.clear();
      gist.reserve(audio.samples.size());
      while(gist.size() < audio.samples.size())
        gist.emplace_back(bufferSize, rate);
    }
  }

  template<auto Func>
  void process(
      const ossia::audio_port& audio,
      ossia::value_port& out_port,
      const ossia::token_request& tk,
      const ossia::exec_state_facade& e)
  {
    preprocess(audio);
    auto it = output.begin();
    auto git = gist.begin();
    for(auto& channel : audio.samples)
    {
      if(bufferSize == channel.size())
      {
        git->processAudioFrame(channel.data(), channel.size());
        *it = float(((*git).*Func)());
      }
      else
      {
        *it = 0.f;
      }
      ++it;
      ++git;
    }
    out_port.write_value(out_val, e.physical_start(tk));
  }

  template<auto Func>
  void process(
      const ossia::audio_port& audio,
      float gain,
      ossia::value_port& out_port,
      const ossia::token_request& tk,
      const ossia::exec_state_facade& e)
  {
    preprocess(audio);
    auto it = output.begin();
    auto git = gist.begin();
    for(auto& channel : audio.samples)
    {
      if(bufferSize == channel.size())
      {
        git->processAudioFrame(channel.data(), channel.size());
        *it = float((git->*Func)()) * gain;
      }
      else
      {
        *it = 0.f;
      }
      ++it;
      ++git;
    }
    out_port.write_value(out_val, e.physical_start(tk));
  }

  template<auto Func>
  void process(
      const ossia::audio_port& audio,
      float gain,
      float gate,
      ossia::value_port& out_port,
      const ossia::token_request& tk,
      const ossia::exec_state_facade& e)
  {
    preprocess(audio);
    auto it = output.begin();
    auto git = gist.begin();
    for(auto& channel : audio.samples)
    {
      if(bufferSize == channel.size())
      {
        git->processAudioFrame(channel.data(), channel.size());
        *it = float(((*git).*Func)()) * gain;
        if(*it < gate)
          *it = 0.f;
      }
      else
      {
        *it = 0.f;
      }
      ++it;
      ++git;
    }
    out_port.write_value(out_val, e.physical_start(tk));
  }

  template<auto Func>
  void processVector(
      const ossia::audio_port& audio,
      ossia::audio_port& mfcc,
      const ossia::token_request& tk,
      const ossia::exec_state_facade& e)
  {
    while(gist.size() < audio.samples.size())
      gist.emplace_back(bufferSize, rate);

    mfcc.samples.resize(audio.samples.size());
    auto it = mfcc.samples.begin();
    auto git = gist.begin();
    for(auto& channel : audio.samples)
    {
      if(bufferSize == channel.size())
      {
        git->processAudioFrame(channel.data(), channel.size());

        auto& res = ((*git).*Func)();
        it->assign(res.begin(), res.end());
      }
      else
      {
        it->clear();
      }

      ++it;
      ++git;
    }
  }

  ossia::small_vector<Gist<double>, 2> gist;
  ossia::value out_val;
  std::vector<ossia::value>& output;
  int bufferSize{};
  int rate{};
};
}
