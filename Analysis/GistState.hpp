#pragma once
#include <Audio/Settings/Model.hpp>
#include <Gist.h>
#include <ossia/dataflow/audio_port.hpp>
#include <ossia/dataflow/graph_node.hpp>
#include <ossia/dataflow/token_request.hpp>
#include <ossia/dataflow/value_port.hpp>
#include <ossia/network/value/value.hpp>

namespace Analysis
{
struct GistState
{
  // For efficiency we take a reference to the vector<value> member
  // of the ossia variant
  explicit GistState(Audio::Settings::Model& settings):
    gist{settings.getBufferSize(), settings.getRate()}
  , out_val{std::vector<ossia::value>{}}
  , output{out_val.v.m_impl.m_value8}
  , bufferSize{settings.getBufferSize()}
  {
    out_val = output;
  }

  explicit GistState():
    GistState{score::AppContext().settings<Audio::Settings::Model>()}
  {
  }

  template<auto Func>
  void process(
      const ossia::audio_port& audio,
      ossia::value_port& out_port,
      const ossia::token_request& tk,
      const ossia::exec_state_facade& e)
  {
    output.resize(audio.samples.size());
    auto it = output.begin();
    for(auto& channel : audio.samples)
    {
      if(bufferSize == channel.size())
      {
        gist.processAudioFrame(channel.data(), channel.size());
        *it = float((gist.*Func)());
      }
      else
      {
        *it = 0.f;
      }
      ++it;
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
    output.resize(audio.samples.size());
    auto it = output.begin();
    for(auto& channel : audio.samples)
    {
      if(bufferSize == channel.size())
      {
        gist.processAudioFrame(channel.data(), channel.size());
        *it = float((gist.*Func)()) * gain;
      }
      else
      {
        *it = 0.f;
      }
      ++it;
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
    mfcc.samples.resize(audio.samples.size());
    auto it = mfcc.samples.begin();
    for(auto& channel : audio.samples)
    {
      if(bufferSize == channel.size())
      {
        gist.processAudioFrame(channel.data(), channel.size());

        auto& res = (gist.*Func)();
        it->assign(res.begin(), res.end());
      }
      else
      {
        it->clear();
      }

      ++it;
    }
  }

  Gist<double> gist;
  ossia::value out_val;
  std::vector<ossia::value>& output;
  int bufferSize{};
};
}
