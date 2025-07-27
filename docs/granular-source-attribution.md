# Granular Source Attribution Analysis & Implementation Plan

## Problem Statement

The current Care-GraphRAG system provides only page-level source attribution, which is insufficient for clinical decision-making. Healthcare professionals need paragraph-level precision to quickly verify specific clinical recommendations.

## Current Limitations

### 1. Source Attribution Issues
- **Page-level only**: `https://cks.nice.org.uk/topics/hypertension/management/management/`
- **No section anchors**: Missing precise location within pages
- **No paragraph IDs**: Cannot point to specific clinical recommendations
- **Graph search artifacts**: Shows `source_url: direct_graph_search` instead of real URLs

### 2. Chunk Granularity Problems
- **22,681 character chunks** - entire NICE sections, not paragraphs
- **325 lines per chunk** - massive blocks of text
- **No paragraph preservation** - content is aggregated, not segmented
- **Graph entity references** - pointing to whole documents, not specific content

### 3. Real-World Example
**Question**: "What is the first line antihypertensive choice for stage 1 hypertension in a black patient?"

**Current Response**: Points to entire management page
**Needed Response**: Direct link to "Step 1 treatment" bullet point

## NICE Website Structure Analysis

### Confirmed Working Anchors
- ✅ `#antihypertensive-drug-treatment` - Section-level anchor
- ✅ `/html/body/div[2]/div[1]/main/div[1]/div[2]/div[3]/section/section[2]/div/ul[1]/li[2]/strong` - XPath to "Step 1 treatment"

### Non-Existent Anchors
- ❌ `#first-line-treatment` - Does not exist
- ❌ Paragraph-level IDs for treatment steps
- ❌ Granular deep-linking anchors

### Content Structure
```html
Section: "How should I prescribe antihypertensive drug treatment?"
├── General principles
├── Step 1 treatment ← TARGET CONTENT
│   └── "Offer an ACE inhibitor or an ARB first line for people who..."
├── Step 2 treatment
└── Step 3 treatment
```

## Technical Solutions

### Option 1: Text Fragment Linking (Recommended)
Use Chrome's scroll-to-text feature:
```
https://cks.nice.org.uk/topics/hypertension/management/management/#antihypertensive-drug-treatment:~:text=Step%201%20treatment
```

**Advantages:**
- Works in modern browsers
- No server-side changes required to NICE
- Highlights exact content
- Fallback to section anchor

### Option 2: Enhanced Chunking + Context
Store hierarchical context with each chunk:
```json
{
  "content": "Step 1 treatment: Offer an ACE inhibitor...",
  "source_url": "https://cks.nice.org.uk/topics/hypertension/management/management/",
  "section_anchor": "#antihypertensive-drug-treatment",
  "section_title": "How should I prescribe antihypertensive drug treatment?",
  "subsection": "Step 1 treatment",
  "xpath": "/html/body/div[2]/div[1]/main/div[1]/div[2]/div[3]/section/section[2]/div/ul[1]/li[2]",
  "content_hierarchy": ["General principles", "Step 1 treatment"]
}
```

### Option 3: Combined Approach
Generate rich source attribution:
```
Source: NICE CKS Hypertension Management
Direct Link: https://cks.nice.org.uk/topics/hypertension/management/management/#antihypertensive-drug-treatment:~:text=Step%201%20treatment
Section: "How should I prescribe antihypertensive drug treatment?"
Subsection: "Step 1 treatment"
Quote: "Offer an ACE inhibitor or an ARB first line for people who are aged under 55 years..."
```

## Implementation Plan

### Phase 1: Documentation & Analysis ✅
- [x] Document current limitations
- [x] Analyze NICE page structure
- [x] Identify technical solutions

### Phase 2: Core Requirements (After API Development - TASK-061)
**Priority**: Implement after TASK-032-035 (API development)

#### 2.1 Enhanced Scraper (`src/scraper.py`)
- Preserve HTML structure during scraping
- Extract and store XPath locations for key content
- Maintain bullet point hierarchy in chunks
- Store section titles and anchors

#### 2.2 Improved Chunking (`src/document_processor.py`)
- Split content at logical boundaries (bullet points, subsections)
- Preserve contextual hierarchy in metadata
- Store navigational breadcrumbs
- Link chunks to specific page sections

#### 2.3 Enhanced Source Attribution (`src/answer_formatter.py`)
- Generate text fragment URLs where possible
- Include section context in citations
- Provide navigation guidance for users
- Format citations with precise location info

#### 2.4 Retrieval System Updates (`src/hybrid_retriever.py`)
- Preserve original URLs in graph search results
- Return specific content sections, not whole documents
- Include precise location metadata in responses

### Phase 3: User Experience Enhancements
#### 3.1 Citation Formatting
```
[1] NICE CKS Hypertension - Step 1 Treatment
    https://cks.nice.org.uk/topics/hypertension/management/management/#antihypertensive-drug-treatment:~:text=Step%201%20treatment
    "Offer an ACE inhibitor or an ARB first line for people who are aged under 55 years..."
```

#### 3.2 Fallback Strategies
- Primary: Text fragment URL
- Secondary: Section anchor + navigation guidance
- Tertiary: Page URL + quoted content

### Phase 4: Validation & Testing
- Test text fragment URLs across browsers
- Validate XPath stability across NICE updates
- Ensure graceful degradation for unsupported browsers

## Implementation Timing

### Recommended Schedule
1. **Complete API development** (TASK-032-035) - **Priority 1**
2. **Frontend basic implementation** (TASK-036-037) - **Priority 2**
3. **Implement granular attribution** (TASK-061) - **Priority 3**
4. Serverless deployment with enhanced attribution

### Rationale
- Core functionality (graph retrieval, QA chain) is working
- API structure should be established first
- This is a UX enhancement that builds on existing retrieval system
- Can be implemented as incremental improvement without breaking changes

## Success Metrics

### Clinical Utility
- Users can navigate to exact recommendation within 1 click
- Source verification time reduced from minutes to seconds
- Increased confidence in system recommendations

### Technical Metrics
- 95% of responses include precise location links
- Text fragment URLs work in 90% of modern browsers
- Fallback to section anchors maintains usability

## Dependencies

### Internal
- Stable API structure (TASK-032-035)
- Working hybrid retrieval system ✅
- Answer formatting pipeline ✅

### External
- NICE website anchor stability
- Browser support for text fragments
- XPath consistency across NICE updates

## Risks & Mitigations

### Risk: NICE HTML Structure Changes
**Mitigation**: Store multiple location strategies (XPath, text patterns, section headers)

### Risk: Text Fragment Browser Support
**Mitigation**: Graceful fallback to section anchors + navigation guidance

### Risk: Performance Impact
**Mitigation**: Implement as optional feature with caching

## Future Enhancements

### Advanced Features
- Visual highlighting of relevant content
- PDF export with embedded links
- Mobile-optimized citation display
- Integration with clinical decision support tools

### Potential Integrations
- FHIR resource linking
- Clinical pathway mapping
- Evidence level indicators
- Confidence scoring for location precision

---

**Status**: Documented, ready for implementation after API development
**Priority**: Medium (after core API functionality)
**Effort**: 2-3 weeks implementation + testing
**Impact**: High clinical utility improvement