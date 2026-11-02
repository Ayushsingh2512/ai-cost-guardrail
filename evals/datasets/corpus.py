"""
Synthetic corpus for the RAG evaluation suite.

This module defines the evaluation corpus as pure Python constants so it is
committed to the repository and deterministic across runs. Every document and
chunk here corresponds to a ``document_id`` used in ``rag_golden.jsonl``.

Pass 1.5 Corpus Hardening
-------------------------
The corpus is expanded from 13 chunks (5 documents) to 47 chunks (18 documents).
- Existing documents and chunks retain identical document IDs and chunk indices
  to ensure all existing expected_evidence references remain valid.
- Distractor documents include:
  * Topically adjacent near-duplicates with different parameters (e.g. EMEA policies,
    executive tiers, contractor guidelines, freight shipping, cloud security) to force
    retrieval discrimination.
  * Clearly unrelated business documents (facilities, IT hardware, code of conduct,
    data governance, IP, procurement) to evaluate noise rejection.
- None of the distractor documents contain facts that answer any currently
  unanswerable golden questions (q_016–q_020 on dev, q_026, q_027, q_029 on test).
- Fictional policies only: no real company or private data.
"""

from __future__ import annotations

#: Version of the synthetic evaluation corpus.
CORPUS_VERSION: str = "1.5.0"

# ---------------------------------------------------------------------------
# Corpus definition
# ---------------------------------------------------------------------------
# Structure: {stable_doc_id: [chunk_0_text, chunk_1_text, ...]}
# Each string is one chunk whose index equals its position in the list.

SYNTHETIC_CORPUS: dict[str, list[str]] = {
    # =======================================================================
    # Original Golden Documents (Pass 1) — IDs and indices preserved
    # =======================================================================
    "synthetic_hr_policy": [
        # chunk_index 0
        (
            "Human Resources Policy — Vacation Days\n\n"
            "Full-time employees receive 15 vacation days per calendar year. "
            "Part-time employees receive vacation days on a pro-rated basis. "
            "Employees may carry over a maximum of 5 unused vacation days to the "
            "following calendar year.  Any remaining unused vacation days above the "
            "5-day carry-over limit are forfeited at year-end."
        ),
        # chunk_index 1
        (
            "Human Resources Policy — Sick Leave\n\n"
            "The company grants each employee 10 sick days per calendar year. "
            "Sick days do not roll over to the following year.  "
            "Unused sick days cannot be cashed out or converted to vacation days. "
            "Employees requiring extended leave beyond 10 days should apply for "
            "short-term disability leave through the HR portal."
        ),
        # chunk_index 2
        (
            "Human Resources Policy — Remote Work\n\n"
            "Employees in eligible roles may work remotely up to 3 days per week, "
            "subject to manager approval.  Remote work arrangements must be agreed "
            "in writing and reviewed quarterly.  Employees working remotely are "
            "expected to be available during standard business hours and to attend "
            "all mandatory in-person meetings."
        ),
    ],
    "synthetic_refund_policy": [
        # chunk_index 0
        (
            "Customer Refund Policy — Digital Products\n\n"
            "Digital products are eligible for a full refund within 14 days of "
            "purchase, provided they have not been downloaded or accessed.  "
            "Once a digital product has been downloaded or accessed it is no "
            "longer eligible for a refund under any circumstances. "
            "Refund requests for digital products must be submitted through the "
            "online customer portal."
        ),
        # chunk_index 1
        (
            "Customer Refund Policy — Physical Goods\n\n"
            "Physical goods purchased in-store are eligible for a full refund "
            "within 30 days of purchase with the original receipt.  "
            "A restocking fee of 10% of the item's purchase price applies to all "
            "returned physical goods.  Items must be returned in their original "
            "packaging and in unused condition."
        ),
        # chunk_index 2
        (
            "Customer Refund Policy — Returns Without a Receipt\n\n"
            "Physical goods may be returned without the original receipt for "
            "store credit only.  Cash refunds are not available for receipt-less "
            "returns.  A 10% restocking fee still applies to all receipt-less "
            "returns.  The store credit issued is valid for 12 months from the "
            "date of the return."
        ),
    ],
    "synthetic_shipping_policy": [
        # chunk_index 0
        (
            "Shipping Policy — International Orders\n\n"
            "International orders are shipped via GlobalShip Express, our "
            "designated international carrier.  Estimated delivery time is 7 to "
            "14 business days from the date of dispatch.  A tracking number is "
            "emailed to the customer upon dispatch.  The company is not "
            "responsible for customs duties or import taxes."
        ),
        # chunk_index 1
        (
            "Shipping Policy — Out-of-Stock Items\n\n"
            "If an item is out of stock at the time of purchase the order is "
            "placed on backorder automatically.  The customer is notified by email "
            "within 24 hours.  Backordered items are shipped as soon as stock "
            "becomes available.  Customers may cancel a backordered item at any "
            "time before it ships."
        ),
        # chunk_index 2
        (
            "Shipping Policy — Domestic Orders\n\n"
            "Domestic orders are shipped via FastRoute Domestic Logistics.  "
            "Orders placed Monday through Thursday are processed the same business "
            "day and shipped within 2 business days.  Orders placed on Fridays are "
            "processed on the following Monday and shipped within 2 business days "
            "of processing.  Standard domestic delivery takes 3 to 5 business days."
        ),
    ],
    "synthetic_security_policy": [
        # chunk_index 0
        (
            "Information Security Policy — Authentication\n\n"
            "Multi-factor authentication (MFA) is required for all corporate "
            "systems including email, VPN, and the internal HR portal.  "
            "Passwords must be at least 12 characters long and include uppercase "
            "letters, lowercase letters, numbers, and symbols.  Passwords must be "
            "rotated every 90 days.  Sharing passwords with colleagues is strictly "
            "prohibited."
        ),
        # chunk_index 1
        (
            "Information Security Policy — Training\n\n"
            "Employees must complete mandatory security awareness training "
            "annually.  New hires must complete the training within their first "
            "30 days of employment.  Completion is tracked by the IT department "
            "and non-compliance may result in disciplinary action.  The training "
            "covers phishing awareness, data handling, and incident reporting."
        ),
    ],
    "synthetic_expense_policy": [
        # chunk_index 0
        (
            "Expense Reimbursement Policy — Submission Deadlines and Approval\n\n"
            "Expense reports must be submitted within 30 days of the business "
            "trip or event.  Reports submitted after this deadline require "
            "VP-level approval and may not be reimbursed.  Single expenses up to "
            "$500 can be reimbursed without CFO approval.  Expenses above $500 "
            "require CFO sign-off before reimbursement is processed."
        ),
        # chunk_index 1
        (
            "Expense Reimbursement Policy — Receipt Requirements and Meal Cap\n\n"
            "Receipts are required for all expenses exceeding $25, including "
            "meals, transportation, accommodation, and conference fees.  Expenses "
            "below $25 may be submitted without a receipt but must include a "
            "written description.  Meal reimbursements are capped at $75 per day.  "
            "Alcohol is not a reimbursable expense under any category."
        ),
    ],

    # =======================================================================
    # Expanded Distractors (Pass 1.5) — Hard / Topically Adjacent
    # =======================================================================
    "synthetic_hr_policy_emea": [
        # chunk_index 0
        (
            "Human Resources Policy (EMEA) — Vacation Days\n\n"
            "Full-time employees in EMEA entities receive 25 statutory vacation "
            "days per calendar year. Part-time employees receive vacation days "
            "pro-rated based on standard weekly hours. Employees may carry over "
            "up to 8 unused vacation days into the following calendar year with "
            "regional director approval. Any additional untaken vacation days "
            "beyond 8 days expire on March 31 of the new calendar year."
        ),
        # chunk_index 1
        (
            "Human Resources Policy (EMEA) — Sickness Absence\n\n"
            "EMEA employees receive up to 15 statutory sick pay days per calendar "
            "year. Self-certification is permitted for absences lasting up to 5 "
            "consecutive business days. Sickness leave does not accrue cash value "
            "and cannot be rolled over to subsequent years. Sickness lasting longer "
            "than 5 consecutive days requires a certified doctor's medical note "
            "submitted to regional HR."
        ),
        # chunk_index 2
        (
            "Human Resources Policy (EMEA) — Hybrid Work Arrangement\n\n"
            "Employees in eligible European office locations may work remotely up "
            "to 2 days per calendar week upon completing their 6-month probationary "
            "period. Hybrid schedules require direct manager consent and quarterly "
            "scheduling alignment. Employees working from home must maintain a "
            "distraction-free environment and reliable high-speed internet."
        ),
    ],
    "synthetic_hr_policy_contractors": [
        # chunk_index 0
        (
            "Human Resources Guidelines — External Professional Services\n\n"
            "Third-party professional consultants, independent contractors, and "
            "contingent workers are not eligible for paid vacation days, personal "
            "days, or paid company holidays. Contractors must record project hours "
            "worked through the external vendor billing system and take unpaid time "
            "off with 5 business days notice to the engagement lead."
        ),
        # chunk_index 1
        (
            "Human Resources Guidelines — Contractor Sickness and Billing\n\n"
            "Independent contractors do not receive paid sick leave or short-term "
            "disability benefits from the company. In the event of illness preventing "
            "services delivery, the contractor must inform the project manager "
            "immediately and invoice only for hours during which active consulting "
            "services were performed."
        ),
    ],
    "synthetic_refund_policy_enterprise": [
        # chunk_index 0
        (
            "Customer Refund Policy — Enterprise Software Subscriptions\n\n"
            "Enterprise annual software licenses and dedicated cloud instances "
            "are non-refundable once license activation keys have been provisioned "
            "in the enterprise portal. Enterprise customers may cancel auto-renewal "
            "by providing 60 days advance written notice prior to the end of the "
            "annual subscription term."
        ),
        # chunk_index 1
        (
            "Customer Refund Policy — Enterprise Appliance Returns\n\n"
            "Enterprise server appliances and edge gateway hardware may be "
            "returned within 45 days of delivery subject to a 15% restocking fee. "
            "Returned hardware must be intact, undamaged, and shipped in factory "
            "packaging with all accessories and tamper-evident seals unbroken."
        ),
        # chunk_index 2
        (
            "Customer Refund Policy — Enterprise RMA Workflow\n\n"
            "All physical hardware returns require an active Return Merchandise "
            "Authorization (RMA) ticket generated by enterprise technical support. "
            "Packages returned without a valid RMA number clearly marked on the "
            "outer shipping label will be rejected at the warehouse dock and "
            "returned at the customer's expense."
        ),
    ],
    "synthetic_shipping_policy_freight": [
        # chunk_index 0
        (
            "Shipping Policy — Ocean Freight and Bulk Consignments\n\n"
            "Heavy freight shipments exceeding 150 kilograms are transported via "
            "PacificSea Marine Logistics. Estimated oceanic transit duration is "
            "25 to 35 business days from vessel departure. Customers receive customs "
            "clearance manifests and container bill of lading tracking within 48 "
            "hours of harbor departure."
        ),
        # chunk_index 1
        (
            "Shipping Policy — Domestic Overnight Expedited\n\n"
            "Urgent domestic consignments requiring priority shipping are dispatched "
            "through SwiftAir Express Couriers. Overnight domestic shipments guarantee "
            "delivery before 10:30 AM the next business day for orders finalized "
            "before 2:00 PM EST. Flat-rate priority dispatch fees apply to all "
            "expedited parcels."
        ),
        # chunk_index 2
        (
            "Shipping Policy — Transit Loss and Damage Claims\n\n"
            "Damage or missing items incurred during transit must be reported within "
            "7 business days of parcel receipt. The company files carrier claims "
            "directly and issues replacement merchandise within 3 business days of "
            "photographic evidence verification."
        ),
    ],
    "synthetic_security_policy_cloud": [
        # chunk_index 0
        (
            "Information Security Policy — Cloud Infrastructure Access\n\n"
            "Access to production cloud environments requires role-based access "
            "control and hardware security token verification (FIDO2 keys). SSH "
            "access to cloud virtual machines is restricted through a bastion "
            "jump host. Administrative bastion sessions automatically terminate "
            "after 60 minutes of inactivity."
        ),
        # chunk_index 1
        (
            "Information Security Policy — Cryptographic Keys and Secrets\n\n"
            "Encryption keys stored in cloud Key Management Services (KMS) must "
            "be configured for automatic rotation every 180 days. Service account "
            "API tokens must not be committed to source code repositories and must "
            "expire within a maximum lifetime of 1 hour."
        ),
        # chunk_index 2
        (
            "Information Security Policy — Vulnerability Management\n\n"
            "Container base images and production server dependencies are scanned "
            "automatically on a weekly cadence. Critical security vulnerabilities "
            "discovered in production infrastructure must be remediated or mitigated "
            "within 48 hours of detection."
        ),
    ],
    "synthetic_expense_policy_executives": [
        # chunk_index 0
        (
            "Expense Reimbursement Policy — Executive Travel and Approvals\n\n"
            "Executive Vice Presidents and C-suite officers must submit corporate "
            "expense reports within 45 days of transaction completion. Business "
            "expenses exceeding $10,000 require Board Audit Committee review and "
            "sign-off. Executive expense reports must be submitted through the "
            "executive concierge portal."
        ),
        # chunk_index 1
        (
            "Expense Reimbursement Policy — Executive Client Hospitality\n\n"
            "Executive business entertainment and prospective client hospitality "
            "dinners are capped at $200 per attendee. Detailed attendee rosters "
            "and business justification summaries must accompany every entertainment "
            "reimbursement filing. Premium ground transportation is reimbursable "
            "for executive client engagements."
        ),
    ],

    # =======================================================================
    # Expanded Distractors (Pass 1.5) — Unrelated Corporate Policies
    # =======================================================================
    "synthetic_vendor_procurement_policy": [
        # chunk_index 0
        (
            "Vendor Procurement Policy — Due Diligence and Compliance\n\n"
            "Prospective vendors providing SaaS applications or data processing "
            "services must submit a valid SOC 2 Type II audit report and complete "
            "the vendor risk questionnaire. Procurement contracts exceeding $25,000 "
            "require legal department review before signature."
        ),
        # chunk_index 1
        (
            "Vendor Procurement Policy — Purchase Order Authorizations\n\n"
            "Department directors may authorize purchase orders up to $20,000. "
            "Expenditures between $20,000 and $100,000 require divisional Vice "
            "President sign-off. Purchase orders exceeding $100,000 require joint "
            "approval from the Chief Financial Officer and Chief Operating Officer."
        ),
        # chunk_index 2
        (
            "Vendor Procurement Policy — Invoicing and Payment Terms\n\n"
            "Standard commercial payment terms for approved suppliers are Net 45 "
            "days from the date of invoice validation by accounts payable. Early "
            "payment discounts of 2% are accepted for settlements completed within "
            "10 calendar days of invoice receipt."
        ),
    ],
    "synthetic_facility_guidelines": [
        # chunk_index 0
        (
            "Facility Management Guidelines — Headquarters Access and Badging\n\n"
            "Headquarters facilities are open to badged employees Monday through "
            "Friday from 6:30 AM to 8:30 PM. All visitors, clients, and maintenance "
            "guests must register at the reception desk, present government photo "
            "identification, and wear a temporary visitor badge while on premises."
        ),
        # chunk_index 1
        (
            "Facility Management Guidelines — Flexible Desk Booking\n\n"
            "Employees utilizing flexible shared workspaces must book hot-desks "
            "via the workplace reservation mobile app at least 12 hours prior to "
            "arrival. Unclaimed desk reservations are released back to the general "
            "pool after 9:30 AM."
        ),
        # chunk_index 2
        (
            "Facility Management Guidelines — Parking and EV Charging\n\n"
            "On-site parking spaces are allocated on a first-come, first-served "
            "basis. Electric vehicle charging stalls are reserved exclusively for "
            "actively charging vehicles with a maximum consecutive plug-in "
            "duration of 4 hours."
        ),
    ],
    "synthetic_it_equipment_policy": [
        # chunk_index 0
        (
            "Information Technology Policy — Hardware Allocation and Refresh\n\n"
            "Standard engineering laptops consist of 16-inch developer workstations "
            "with 32 GB RAM. Non-engineering employees receive lightweight 14-inch "
            "enterprise laptops. Corporate hardware is scheduled for hardware refresh "
            "every 36 months from initial deployment."
        ),
        # chunk_index 1
        (
            "Information Technology Policy — Ergonomic Peripherals Stipend\n\n"
            "New full-time corporate hires may claim a one-time home office peripheral "
            "stipend of up to $300 for ergonomic monitors, keyboards, or external mice "
            "within their first 90 days of employment. Peripheral reimbursement "
            "claims are filed through the IT service catalog."
        ),
        # chunk_index 2
        (
            "Information Technology Policy — Stolen or Lost Equipment Reporting\n\n"
            "Any lost, stolen, or physically damaged corporate computing asset must "
            "be reported to the IT service desk and security operations center within "
            "12 hours of discovery. Remote enterprise wipe commands will be "
            "dispatched immediately to protect corporate data."
        ),
    ],
    "synthetic_social_media_policy": [
        # chunk_index 0
        (
            "Communications Policy — External Inquiries and Spokespersons\n\n"
            "Only designated company spokespersons authorized by the Corporate "
            "Communications team may deliver statements to news media or industry "
            "reporters. Any inbound interview or press inquiry received by an "
            "employee must be forwarded to press@example-corp.internal within 4 hours."
        ),
        # chunk_index 1
        (
            "Communications Policy — Personal Social Network Usage\n\n"
            "Employees discussing industry trends on personal social networking "
            "accounts (such as LinkedIn or personal blogs) must include the following "
            "disclaimer: 'Views and opinions expressed are solely my own and do not "
            "represent the views of my employer.' Confidential company projects must "
            "not be discussed publicly."
        ),
        # chunk_index 2
        (
            "Communications Policy — Social Endorsements and Brand Guidelines\n\n"
            "Employees sharing company product announcements or promotional content "
            "on public platforms must clearly disclose their employment affiliation "
            "using prominent hashtags such as #CompanyTeam. Deceptive or anonymous "
            "testimonials regarding company offerings are strictly prohibited."
        ),
    ],
    "synthetic_code_of_conduct": [
        # chunk_index 0
        (
            "Code of Business Conduct — Workplace Standards\n\n"
            "The company is committed to maintaining a workplace free from "
            "discrimination, sexual harassment, and hostility. Any form of verbal "
            "or physical conduct that demeans an individual based on race, gender, "
            "religion, age, disability, or background is subject to immediate "
            "disciplinary termination."
        ),
        # chunk_index 1
        (
            "Code of Business Conduct — Gifts and Entertainment Limitations\n\n"
            "Employees may not accept business gifts, meals, or promotional "
            "merchandise from current suppliers or prospective vendors exceeding "
            "$50 in estimated value. Promotional tokens of nominal value under $50, "
            "such as branded pens or calendars, may be accepted."
        ),
        # chunk_index 2
        (
            "Code of Business Conduct — Ethics Reporting and Whistleblower Hotline\n\n"
            "Suspected financial fraud, accounting irregularities, or regulatory "
            "non-compliance may be reported confidentially and anonymously through "
            "the 24/7 EthicsPoint whistleblower hotline. The company strictly "
            "prohibits retaliation of any kind against whistleblowers reporting in "
            "good faith."
        ),
    ],
    "synthetic_data_classification_policy": [
        # chunk_index 0
        (
            "Data Governance Policy — Information Classification Levels\n\n"
            "All organizational information assets are classified into one of "
            "four tiers: Public, Internal Business Use, Confidential, and Highly "
            "Restricted. Classification determines required storage encryption, "
            "access authorization levels, and cross-border transfer safeguards."
        ),
        # chunk_index 1
        (
            "Data Governance Policy — Handling Highly Restricted Records\n\n"
            "Highly Restricted data, including customer payment card information "
            "and unmasked government identifiers, must be encrypted with AES-256 "
            "both at rest and in transit. Access is limited strictly to named "
            "individuals with justified business needs and multi-factor approval."
        ),
        # chunk_index 2
        (
            "Data Governance Policy — Document Retention Schedules\n\n"
            "Corporate financial ledger records must be preserved for 7 calendar "
            "years in compliance with statutory audit standards. Routine business "
            "emails and internal messaging chat logs are automatically archived "
            "after 180 days and permanently expunged after 24 months."
        ),
    ],
    "synthetic_intellectual_property_policy": [
        # chunk_index 0
        (
            "Intellectual Property Policy — Inventions and Proprietary Works\n\n"
            "All software source code, system architectures, documentation, and "
            "technical innovations created by employees during working hours or "
            "using company computing assets are the sole and exclusive intellectual "
            "property of the company under work-for-hire principles."
        ),
        # chunk_index 1
        (
            "Intellectual Property Policy — Open Source Contributions\n\n"
            "Employees wishing to release open-source code libraries or contribute "
            "bug fixes to external open-source projects using company equipment "
            "must obtain prior approval from the Open Source Program Office (OSPO) "
            "via an internal contribution ticket."
        ),
        # chunk_index 2
        (
            "Intellectual Property Policy — Trademark Guidelines\n\n"
            "Company trade names, logos, product mascots, and marketing slogans "
            "must not be modified, recolored, or integrated into personal projects "
            "or third-party partner marketing without explicit brand governance "
            "team sign-off."
        ),
    ],
}

# ---------------------------------------------------------------------------
# Derived metadata for validation
# ---------------------------------------------------------------------------

#: Set of all (document_id, chunk_index) pairs present in the corpus.
VALID_EVIDENCE_REFS: frozenset[tuple[str, int]] = frozenset(
    (doc_id, chunk_idx)
    for doc_id, chunks in SYNTHETIC_CORPUS.items()
    for chunk_idx in range(len(chunks))
)

#: All stable document IDs present in the corpus.
CORPUS_DOCUMENT_IDS: frozenset[str] = frozenset(SYNTHETIC_CORPUS.keys())

#: Total count of documents in the synthetic corpus.
CORPUS_DOCUMENT_COUNT: int = len(SYNTHETIC_CORPUS)

#: Total count of chunks in the synthetic corpus.
CORPUS_CHUNK_COUNT: int = len(VALID_EVIDENCE_REFS)
