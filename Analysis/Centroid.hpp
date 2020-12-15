#pragma once
#include <Engine/Node/PdNode.hpp>
#include <Analysis/GistState.hpp>
#include <numeric>
namespace Analysis
{
struct Centroid
{
  struct Metadata : Control::Meta_base
  {
    static const constexpr auto prettyName = "Centroid";
    static const constexpr auto objectKey = "Centroid";
    static const constexpr auto category = "Analysis";
    static const constexpr auto author = "ossia score, Gist library";
    static const constexpr auto kind = Process::ProcessCategory::Analyzer;
    static const constexpr auto description = "Get the centroid of a signal";
    static const constexpr auto tags = std::array<const char*, 0>{};
    static const uuid_constexpr auto uuid = make_uuid("9d26b429-a417-4c98-a4c6-70af90a5c4ab");

    static const constexpr audio_in audio_ins[]{"in"};
    static const constexpr value_out value_outs[]{"out"};
  };

  using State = GistState;
  using control_policy = ossia::safe_nodes::default_tick;

  static void
  run(const ossia::audio_port& in,
      ossia::value_port& out,
      ossia::token_request tk,
      ossia::exec_state_facade e,
      State& st)
  {
    st.process<&Gist<double>::spectralCentroid>(in, out, tk, e);
  }
};
}
