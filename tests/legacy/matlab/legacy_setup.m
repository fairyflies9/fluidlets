function p = legacy_setup()
% Put the legacy code and the CUDA shims on the path and move into a
% scratch work folder, so legacy scripts that write files (towers.txt,
% velocity_t_xyz_final.raw) never touch legacy/.
here = fileparts(mfilename('fullpath'));
p.legacy = fullfile(here, '..', '..', '..', 'legacy', 'sdc_cuda3');
p.data = fullfile(here, '..', 'data');
p.work = fullfile(here, 'work');
addpath(p.legacy);
addpath(fullfile(here, 'shims'));   % added last, so it shadows the .mexw32 names
addpath(here);
if ~exist(p.work, 'dir'), mkdir(p.work); end
if ~exist(p.data, 'dir'), mkdir(p.data); end
cd(p.work);
set(0, 'DefaultFigureVisible', 'off');
end
