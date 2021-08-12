#pragma once
#include <Media/Commands/ChangeAudioFile.hpp>
#include <Media/Sound/SoundModel.hpp>
#include <Process/OfflineAction/OfflineAction.hpp>

#include <score/command/Dispatchers/CommandDispatcher.hpp>
#include <score/document/DocumentContext.hpp>

#include <core/document/Document.hpp>

#include <ossia/audio/drwav_handle.hpp>

#include <QDir>
namespace VariableMarkovOracleCentroid
{

class GainDoubler final : public Process::OfflineAction
{
  SCORE_CONCRETE("05145365-9646-43B8-9F21-23E3762AEDA6")

  QString title() const noexcept override;
  UuidKey<Process::ProcessModel> target() const noexcept override;
  void
  apply(Process::ProcessModel& proc, const score::DocumentContext&) override;
};
}