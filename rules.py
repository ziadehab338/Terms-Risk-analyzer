RISK_RULES = {
    "Data Sharing": {
        "keywords": [
            "share your data",
            "share your information",
            "third party",
            "third-party",
            "affiliates",
            "partners",
            "disclose your information",
            "sell your data",
            "allow third parties to access your data",
            "share with advertisers",
            "share with service providers",
            "share with business partners",
            "share with affiliates",
            "share with third-party service providers",
            "allow access to your data ",
            "share your personal information",
            "allow  access to storage of your data",
            "track your data",
            "share your location data",
            

        ],
        "risk_level": "High",
        "weight": 3,
        "explanation": "This clause may allow sharing user data with external parties."
    },

    "Payment / Auto-Renewal": {
        "keywords": [
            "auto-renew",
            "automatically renew",
            "recurring billing",
            "subscription fee",
            "charged automatically",
            "billing cycle",
            "payment method",
            "additional charges",
            "save your payment information",
            "transaction fees",
            "refund policy",
            "cancellation policy",
            "cancel your subscription",
            "renewal terms",
            "billing terms",
            "payment terms",
            "subscription terms",
            "automatic payment",
            "automatic billing",
            "automatic renewal",
            "renewal fee",
            "renewal charge",
            "renewal policy",
            "refund policy",
            "cancellation policy",

        ],
        "risk_level": "High",
        "weight": 3,
        "explanation": "This clause may involve automatic charges or subscription renewal."
    },

    "Privacy / Tracking": {
        "keywords": [
            "collect your location",
            "track your activity",
            "usage data",
            "device information",
            "location information",
            "monitor your behavior",
            "tracking technologies",
            "cookies",
            "location tracking"
            "allow advertisers to track you"
            "collect information about your usage",
            "app to track your location",
            "app to track user behavior",

        ],
        "risk_level": "Medium",
        "weight": 2,
        "explanation": "This clause may involve user tracking or collection of personal data."
    },

    "Permissions Access": {
        "keywords": [
            "access your camera",
            "access your microphone",
            "access your contacts",
            "access your files",
            "camera",
            "microphone",
            "contacts",
            "storage permission",
            "allow access to your device's permissions",
            "allow access to your device's camera",
            "allow access to your device's microphone",
        ],
        "risk_level": "High",
        "weight": 3,
        "explanation": "This clause may request access to sensitive device permissions."
    },

    "Legal Liability": {
        "keywords": [
            "not liable",
            "no liability",
            "limitation of liability",
            "waive your rights",
            "disclaim all warranties",
            "as is",
            "at your own risk"
        ],
        "risk_level": "Medium",
        "weight": 2,
        "explanation": "This clause may reduce the company's legal responsibility."
    },

    "Account Termination": {
        "keywords": [
            "terminate your account",
            "suspend your account",
            "remove your access",
            "without notice",
            "at any time"
        ],
        "risk_level": "Medium",
        "weight": 2,
        "explanation": "This clause may allow the service to suspend or terminate access unexpectedly."
    }
}