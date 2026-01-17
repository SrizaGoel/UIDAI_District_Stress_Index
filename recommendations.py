import pandas as pd

def recommend_action(row):
    """Generate actionable recommendations based on multiple factors"""
    
    recommendations = []
    
    # Base on persistent stress
    if row["chronic_stress"]:
        recommendations.append("🚨 **Immediate Intervention Required**: Deploy permanent additional staff and infrastructure")
        recommendations.append("📋 **Action**: Establish dedicated Aadhaar center with extended hours")
    elif row["persistent_stress"]:
        recommendations.append("⚠️ **Persistent Issue**: Increase staffing by 50% for next 3 months")
        recommendations.append("📋 **Action**: Temporary staff augmentation and mobile unit deployment")
    
    # Base on priority level
    if row["priority_level"] == "CRITICAL":
        recommendations.append("🔥 **Critical Priority**: Deploy 2+ mobile units immediately")
        recommendations.append("⏰ **Action**: Extend working hours by 4 hours daily")
        recommendations.append("📞 **Action**: Set up helpline for appointment scheduling")
    elif row["priority_level"] == "WATCH":
        recommendations.append("👀 **Monitor Closely**: Weekly review required")
        recommendations.append("📋 **Action**: Prepare contingency plan for staff reallocation")
    
    # Base on stress level
    if row["stress_level"] == "HIGH":
        recommendations.append("🔴 **High Stress**: Temporary staff reallocation needed")
        recommendations.append("📋 **Action**: Reallocate 30% staff from nearby low-stress districts")
    elif row["stress_level"] == "MEDIUM":
        recommendations.append("🟡 **Medium Stress**: Enhanced monitoring")
        recommendations.append("📋 **Action**: Bi-weekly progress review")
    
    # Additional recommendations based on other factors
    if row.get("biometric_penetration", 0) < 0.3:
        recommendations.append("📱 **Low Biometric Coverage**: Run awareness campaigns")
    
    if row.get("demographic_coverage", 0) < 0.4:
        recommendations.append("👥 **Documentation Issues**: Simplify document submission process")
    
    if row["emergency_flag"]:
        recommendations.append("🚑 **Emergency Response**: Activate district-level emergency protocol")
        recommendations.append("📋 **Action**: Daily progress reporting to state headquarters")
    
    if not recommendations:
        recommendations.append("✅ **Normal Operations**: Continue regular monitoring")
        recommendations.append("📋 **Action**: Monthly review meetings")
    
    return "<br>".join(recommendations)

def generate_resource_plan(df):
    """Generate resource allocation plan"""
    
    resource_plan = {}
    
    high_stress_districts = df[df["stress_level"] == "HIGH"]
    critical_districts = df[df["priority_level"] == "CRITICAL"]
    
    resource_plan["mobile_units_needed"] = len(critical_districts) * 2
    resource_plan["additional_staff"] = len(high_stress_districts) * 5
    resource_plan["priority_districts"] = critical_districts[["state", "district"]].to_dict('records')
    
    return resource_plan