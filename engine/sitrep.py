"""
NSG Tactical AI Command Engine - Automated MHA/NSG Standard SITREP Generator
Generates military Situation Reports with threat matrices and tactical directives.
"""
import time
import json
from typing import List, Dict


class SITREPGenerator:
    @staticmethod
    def generate(
        operation_name: str = "OPERATION SAGAR DEFENSE",
        icp_node: str = "ICP-DELTA-01 (HQ)",
        sector: str = "SECTOR 4 - PERIMETER WEST",
        active_profiles: List[Dict] = None,
        incident_log: List[Dict] = None,
        active_zones: List[Dict] = None
    ) -> Dict:
        active_profiles = active_profiles or []
        incident_log = incident_log or []
        active_zones = active_zones or []

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        total_targets = len(active_profiles)

        critical_count = sum(1 for p in active_profiles if p.get("risk_level") == "CRITICAL" or p.get("in_restricted_zone") == 1)
        high_count = sum(1 for p in active_profiles if p.get("risk_level") == "HIGH")
        medium_count = sum(1 for p in active_profiles if p.get("risk_level") == "MEDIUM")

        # Threat Level Determination
        if critical_count > 0:
            overall_threat = "RED (CRITICAL - HOSTILE BREACH DETECTED)"
        elif high_count > 0:
            overall_threat = "AMBER (HIGH - ERRATIC KINEMATICS / ANOMALY)"
        elif medium_count > 0:
            overall_threat = "YELLOW (ELEVATED - LOITERING SUSPECTS)"
        else:
            overall_threat = "GREEN (NOMINAL SECTOR STATUS)"

        # Tactical Actionable Directives
        directives = []
        if critical_count > 0:
            directives.append("CRITICAL DIRECTIVE: Deploy NSG Rapid Action Hit Team to Alpha Sector immediately for suspect interception.")
        
        has_bag_threat = any(item.get("type") == "UNATTENDED_BAGGAGE_IED" for item in incident_log)
        if has_bag_threat:
            directives.append("EOD DISPATCH: Dispatch Bomb Disposal Squad (BDS / EOD) with K9 unit to secure 50m radius around unattended baggage.")

        has_panic = any(item.get("type") == "CROWD_DISPERSAL_PANIC" for item in incident_log)
        if has_panic:
            directives.append("TACTICAL CAUTION: Possible ambush or detonation event detected. Seal exit choke points and maintain perimeter containment.")

        if not directives:
            directives.append("ROUTINE: Maintain optical and thermal UAV drone overwatch. Continue standard perimeter patrols.")

        # Generate Markdown
        markdown_lines = [
            "# NATIONAL SECURITY GUARD (NSG) - TACTICAL SITREP",
            "**MINISTRY OF HOME AFFAIRS (MHA) - SPECIAL OPERATIONS COMMAND**",
            "---",
            f"- **OPERATION:** {operation_name}",
            f"- **INCIDENT COMMAND POST:** {icp_node}",
            f"- **SECTOR:** {sector}",
            f"- **TIMESTAMP:** {timestamp}",
            f"- **THREAT ADVISORY LEVEL:** **{overall_threat}**",
            "",
            "---",
            "",
            "### 1. EXECUTIVE SUMMARY & TARGET TELEMETRY",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Total Tracked Targets** | {total_targets} |",
            f"| **Critical Threats (Breaches)** | {critical_count} |",
            f"| **High Risk Targets (Anomalous)** | {high_count} |",
            f"| **Medium Risk Targets (Loiterers)** | {medium_count} |",
            f"| **Active Geofence Boundaries** | {len(active_zones)} |",
            f"| **Total Incidents Logged** | {len(incident_log)} |",
            "",
            "---",
            "",
            "### 2. ACTIONABLE COMMAND DIRECTIVES FOR HIT TEAMS"
        ]

        for idx, d in enumerate(directives, 1):
            markdown_lines.append(f"{idx}. {d}")

        markdown_lines.extend([
            "",
            "---",
            "",
            "### 3. CHRONOLOGICAL INCIDENT LOG"
        ])

        if incident_log:
            markdown_lines.append("| Timestamp | Severity | Threat Category | Incident Details |")
            markdown_lines.append("| :--- | :--- | :--- | :--- |")
            for inc in incident_log[-15:]:
                t_stamp = inc.get("timestamp", "00:00")
                sev = inc.get("severity", "HIGH")
                cat = inc.get("type", "ALERT")
                msg = inc.get("message", "No details")
                markdown_lines.append(f"| {t_stamp} | **{sev}** | `{cat}` | {msg} |")
        else:
            markdown_lines.append("*No high-priority tactical incidents recorded in this session.*")

        markdown_lines.extend([
            "",
            "---",
            "*CLASSIFIED // NSG TACTICAL ICP SYSTEM GENERATED REPORT*"
        ])

        markdown = "\n".join(markdown_lines)

        return {
            "operation": operation_name,
            "icp": icp_node,
            "sector": sector,
            "timestamp": timestamp,
            "threat_level": overall_threat,
            "metrics": {
                "total_targets": total_targets,
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "total_incidents": len(incident_log)
            },
            "directives": directives,
            "markdown": markdown
        }
