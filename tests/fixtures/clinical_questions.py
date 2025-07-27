"""
Clinical questions and expected answers for testing.
TASK-028: Sample questions/answers for validation dataset.
"""

from typing import Dict, List, Any

# Clinical questions with expected answers based on NICE CKS Hypertension guidelines
CLINICAL_QUESTIONS = [
    {
        "id": "q001",
        "question": "What is the first-line treatment for hypertension in a 45-year-old white patient?",
        "category": "age_specific_treatment",
        "expected_answer": {
            "main_answer": "ACE inhibitor or ARB (angiotensin receptor blocker)",
            "key_points": [
                "For patients under 55 years who are not of African or Caribbean descent",
                "ACE inhibitor is first choice, ARB if ACE inhibitor not tolerated",
                "Start with lowest effective dose"
            ],
            "confidence": "high",
            "clinical_safety": "safe"
        },
        "source_sections": [
            "Treatment pathway",
            "First-line treatment options",
            "Age-specific recommendations"
        ]
    },
    {
        "id": "q002", 
        "question": "What is the first-line treatment for hypertension in a 60-year-old patient?",
        "category": "age_specific_treatment",
        "expected_answer": {
            "main_answer": "Calcium channel blocker (CCB)",
            "key_points": [
                "For patients 55 years and over regardless of ethnicity",
                "Thiazide-like diuretic if CCB not suitable",
                "Consider amlodipine as preferred CCB"
            ],
            "confidence": "high",
            "clinical_safety": "safe"
        },
        "source_sections": [
            "Treatment pathway",
            "First-line treatment options",
            "Age-specific recommendations"
        ]
    },
    {
        "id": "q003",
        "question": "What blood pressure target should be aimed for in a 70-year-old with hypertension?",
        "category": "blood_pressure_targets",
        "expected_answer": {
            "main_answer": "Less than 140/90 mmHg",
            "key_points": [
                "Standard target for most adults with hypertension",
                "Lower target of 130/80 mmHg if high cardiovascular risk",
                "Measure blood pressure in both arms initially"
            ],
            "confidence": "high",
            "clinical_safety": "safe"
        },
        "source_sections": [
            "Blood pressure targets",
            "Monitoring and review",
            "Risk assessment"
        ]
    },
    {
        "id": "q004",
        "question": "When should ambulatory blood pressure monitoring be offered?",
        "category": "diagnosis_monitoring",
        "expected_answer": {
            "main_answer": "When clinic blood pressure is 140/90 mmHg or higher",
            "key_points": [
                "To confirm diagnosis of hypertension",
                "24-hour or home blood pressure monitoring",
                "Average of at least 14 measurements taken over 2-7 days",
                "Exclude white coat hypertension"
            ],
            "confidence": "high", 
            "clinical_safety": "safe"
        },
        "source_sections": [
            "Diagnosis",
            "Blood pressure measurement",
            "Ambulatory monitoring"
        ]
    },
    {
        "id": "q005",
        "question": "What lifestyle advice should be given to patients with hypertension?",
        "category": "lifestyle_management", 
        "expected_answer": {
            "main_answer": "Diet, exercise, weight management, alcohol reduction, and smoking cessation",
            "key_points": [
                "Reduce salt intake to less than 6g per day",
                "Regular aerobic exercise (at least 30 minutes, 5 days per week)",
                "Maintain healthy weight (BMI 20-25)",
                "Limit alcohol intake",
                "Stop smoking if applicable"
            ],
            "confidence": "high",
            "clinical_safety": "safe"
        },
        "source_sections": [
            "Lifestyle management",
            "Non-pharmacological treatment",
            "Prevention advice"
        ]
    },
    {
        "id": "q006",
        "question": "What is the second-line treatment if ACE inhibitor alone is insufficient?",
        "category": "combination_therapy",
        "expected_answer": {
            "main_answer": "Add calcium channel blocker or thiazide-like diuretic",
            "key_points": [
                "Combination therapy with ACE inhibitor plus CCB or diuretic",
                "Fixed-dose combinations available to improve adherence",
                "Monitor for hypotension with combination therapy"
            ],
            "confidence": "high",
            "clinical_safety": "safe"
        },
        "source_sections": [
            "Step 2 treatment",
            "Combination therapy",
            "Treatment escalation"
        ]
    },
    {
        "id": "q007",
        "question": "Are there any contraindications to ACE inhibitors?",
        "category": "contraindications_cautions",
        "expected_answer": {
            "main_answer": "Yes, including pregnancy, bilateral renal artery stenosis, and severe aortic stenosis",
            "key_points": [
                "Pregnancy and breastfeeding",
                "Bilateral renal artery stenosis",
                "Severe aortic stenosis",
                "Previous angioedema with ACE inhibitor",
                "Monitor renal function and electrolytes"
            ],
            "confidence": "high",
            "clinical_safety": "critical"
        },
        "source_sections": [
            "Contraindications",
            "Drug safety",
            "Prescribing considerations"
        ]
    },
    {
        "id": "q008",
        "question": "How often should blood pressure be monitored once treatment is started?",
        "category": "monitoring_frequency",
        "expected_answer": {
            "main_answer": "Every 4-6 weeks until target achieved, then annually",
            "key_points": [
                "Monitor every 4-6 weeks when adjusting treatment",
                "Once stable and at target, annual review",
                "More frequent monitoring if cardiovascular risk factors",
                "Check renal function and electrolytes with ACE inhibitors/ARBs"
            ],
            "confidence": "high",
            "clinical_safety": "safe"
        },
        "source_sections": [
            "Monitoring and follow-up",
            "Treatment review",
            "Ongoing care"
        ]
    },
    {
        "id": "q009",
        "question": "What are the symptoms of malignant hypertension?",
        "category": "emergency_complications", 
        "expected_answer": {
            "main_answer": "Severe headache, visual disturbance, chest pain, breathlessness, and neurological symptoms",
            "key_points": [
                "Blood pressure typically >180/120 mmHg with end-organ damage",
                "Papilloedema on fundoscopy",
                "Acute kidney injury",
                "Hypertensive encephalopathy",
                "Requires immediate specialist referral"
            ],
            "confidence": "high",
            "clinical_safety": "critical"
        },
        "source_sections": [
            "Complications",
            "Emergency presentation", 
            "Malignant hypertension",
            "Red flag symptoms"
        ]
    },
    {
        "id": "q010",
        "question": "Should patients with diabetes have different blood pressure targets?",
        "category": "comorbidity_management",
        "expected_answer": {
            "main_answer": "Yes, lower target of 130/80 mmHg for patients with diabetes",
            "key_points": [
                "Target 130/80 mmHg rather than 140/90 mmHg",
                "Higher cardiovascular risk requires tighter control",
                "ACE inhibitor or ARB preferred first-line",
                "Regular monitoring of renal function"
            ],
            "confidence": "high",
            "clinical_safety": "safe"
        },
        "source_sections": [
            "Diabetes and hypertension",
            "Comorbidity management",
            "Cardiovascular risk reduction"
        ]
    }
]

# Edge cases and challenging scenarios
EDGE_CASE_QUESTIONS = [
    {
        "id": "edge001",
        "question": "What should I do if my patient is allergic to both ACE inhibitors and ARBs?",
        "category": "complex_prescribing",
        "expected_answer": {
            "main_answer": "Use calcium channel blocker or thiazide-like diuretic as first-line",
            "key_points": [
                "CCB preferred if under 55 and not African/Caribbean descent",
                "Thiazide-like diuretic alternative option",
                "Document allergy clearly",
                "Consider specialist referral for complex cases"
            ],
            "confidence": "medium",
            "clinical_safety": "safe"
        }
    },
    {
        "id": "edge002", 
        "question": "Can I use herbal remedies alongside blood pressure medication?",
        "category": "complementary_medicine",
        "expected_answer": {
            "main_answer": "Caution advised - some herbal remedies can interact with blood pressure medications",
            "key_points": [
                "Discuss all supplements with healthcare provider",
                "Some herbs can affect blood pressure",
                "Potential drug interactions",
                "Do not stop prescribed medication"
            ],
            "confidence": "medium",
            "clinical_safety": "caution"
        }
    }
]

# Questions that should return low confidence or "insufficient information"
INSUFFICIENT_INFORMATION_QUESTIONS = [
    {
        "id": "insuff001",
        "question": "What is the exact mechanism of action of lisinopril at the molecular level?",
        "category": "detailed_pharmacology",
        "expected_response": "insufficient_information",
        "reason": "NICE CKS focuses on clinical guidance, not detailed pharmacology"
    },
    {
        "id": "insuff002", 
        "question": "What are the specific blood pressure guidelines for patients on Mars?",
        "category": "irrelevant_context",
        "expected_response": "insufficient_information", 
        "reason": "Question not related to UK clinical guidelines"
    }
]

# Export combined question sets
ALL_TEST_QUESTIONS = CLINICAL_QUESTIONS + EDGE_CASE_QUESTIONS + INSUFFICIENT_INFORMATION_QUESTIONS