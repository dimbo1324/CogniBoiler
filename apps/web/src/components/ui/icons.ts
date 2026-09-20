// The console's icon vocabulary: one meaning, one glyph, chosen once here.
//
// Screens import a meaning (`AlarmsIcon`), never a glyph name, so the same idea cannot end
// up drawn two different ways on two screens. Each export is a plain reference, so a glyph
// nothing uses is dropped from the bundle.

import {
  Activity,
  BellRing,
  ChartLine,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  CircleX,
  ClipboardList,
  Droplets,
  Flame,
  Gauge,
  Hand,
  HeartPulse,
  LayoutDashboard,
  LoaderCircle,
  LogOut,
  Leaf,
  Monitor,
  MonitorCog,
  Moon,
  OctagonX,
  Pause,
  Percent,
  Play,
  Power,
  RotateCcw,
  ScrollText,
  Server,
  SlidersHorizontal,
  Sun,
  Thermometer,
  TriangleAlert,
  Users,
  VolumeX,
  Wifi,
  WifiOff,
  Zap,
} from "lucide-react";

export const BrandIcon = Flame;

// Navigation, in the order of the shell's menu.
export const OverviewIcon = LayoutDashboard;
export const TrendsIcon = ChartLine;
export const AlarmsIcon = BellRing;
export const ControlIcon = SlidersHorizontal;
export const EngineerIcon = MonitorCog;
export const AuditIcon = ScrollText;
export const UsersIcon = Users;
export const PlatformIcon = Server;

// Session and appearance.
export const SignOutIcon = LogOut;
export const ThemeDarkIcon = Moon;
export const ThemeLightIcon = Sun;
export const ThemeSystemIcon = Monitor;

// The live connection to the gateway.
export const LiveIcon = Wifi;
export const OfflineIcon = WifiOff;
export const PendingIcon = LoaderCircle;

// Outcomes and severity.
export const OkIcon = CircleCheck;
export const RefusedIcon = CircleX;
export const WarningIcon = TriangleAlert;
export const CriticalIcon = OctagonX;
export const InfoIcon = CircleAlert;

// The plant and the PLC.
export const PlcIcon = Activity;
export const EmergencyStopIcon = Hand;
export const PowerIcon = Zap;
export const PressureIcon = Gauge;
export const TemperatureIcon = Thermometer;
export const WaterIcon = Droplets;
export const FuelIcon = Flame;
export const EventsIcon = ClipboardList;
export const EfficiencyIcon = Percent;
export const EmissionsIcon = Leaf;
export const HealthIcon = HeartPulse;
export const ResetIcon = RotateCcw;
export const RunIcon = Play;
export const PauseIcon = Pause;
export const UnitIcon = Power;

// Paging and silencing.
export const PreviousIcon = ChevronLeft;
export const NextIcon = ChevronRight;
export const SilenceIcon = VolumeX;
