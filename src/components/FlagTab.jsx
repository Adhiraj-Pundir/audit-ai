import React, { useState } from 'react'
import { Search, AlertCircle, CheckCircle2, Loader2, Info } from 'lucide-react'

const FlagTab = () => {
  const [inputText, setInputText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleAudit = async () => {
    if (!inputText.trim()) return

    setLoading(true)
    setResult(null)
    setError(null)

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${apiUrl}/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          application_text: inputText,
          mitigation: 'none'
        })
      })

      const data = await response.json()
      
      if (response.status === 503) {
        setResult(data.stub_response)
        setError("Model is currently loading. Displaying stub results.")
      } else if (!response.ok) {
        throw new Error(data.message || "Failed to score application")
      } else {
        setResult(data)
      }
    } catch (err) {
      console.error("Score error:", err)
      setError("Backend connection failed. Please ensure the server is running.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fade-in">
      <div className="header-section">
        <h1 className="headline-md">Real-time Bias Flagging</h1>
        <p className="body-base">Audit individual applications for immediate bias detection and probe triggering.</p>
      </div>

      <div className="card">
        <label className="body-base" style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>Loan Application Content</label>
        <textarea 
          className="input-field" 
          rows="6" 
          placeholder="Paste loan application text here..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        ></textarea>
        
        <button 
          className="button-primary" 
          onClick={handleAudit}
          disabled={loading || !inputText.trim()}
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          {loading ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
          {loading ? 'Analyzing...' : 'Run Audit'}
        </button>
      </div>

      {error && (
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', color: 'var(--axis-dalit)', marginBottom: '16px', background: 'rgba(239,68,68,0.1)', padding: '12px', borderRadius: '4px' }}>
          <Info size={18} />
          <span className="body-base" style={{ fontSize: '13px' }}>{error}</span>
        </div>
      )}

      {result && (
        <div className="card fade-in" style={{ borderColor: result.bias_flag ? 'var(--axis-dalit)' : '#10B981' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
            <div>
              <h3 className="headline-md" style={{ fontSize: '18px' }}>Audit Result</h3>
              <p className="body-base">Model Decision: <strong>{result.decision}</strong></p>
            </div>
            {result.bias_flag ? (
              <div style={{ color: 'var(--axis-dalit)', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600 }}>
                <AlertCircle size={20} />
                BIAS FLAG
              </div>
            ) : (
              <div style={{ color: '#10B981', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600 }}>
                <CheckCircle2 size={20} />
                CLEAN
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
            <div style={{ background: 'var(--background)', padding: '16px', borderRadius: 'var(--radius-sm)' }}>
              <div className="body-base" style={{ fontSize: '12px', textTransform: 'uppercase', marginBottom: '4px' }}>Decision Margin</div>
              <div style={{ fontSize: '32px', fontWeight: 700, color: result.margin < 0 ? 'var(--axis-dalit)' : 'inherit' }}>
                {result.margin.toFixed(3)}
              </div>
              <div className="body-base" style={{ fontSize: '11px', marginTop: '4px' }}>
                Approved: {result.approved_logit.toFixed(2)} | Rejected: {result.rejected_logit.toFixed(2)}
              </div>
            </div>

            <div style={{ background: 'var(--background)', padding: '16px', borderRadius: 'var(--radius-sm)' }}>
              <div className="body-base" style={{ fontSize: '12px', textTransform: 'uppercase', marginBottom: '4px' }}>Max Counterfactual Delta</div>
              <div style={{ fontSize: '24px', fontWeight: 600, color: result.counterfactual_probe?.max_delta > 0.5 ? 'var(--axis-dalit)' : 'inherit' }}>
                {result.counterfactual_probe?.max_delta.toFixed(3)}
              </div>
              <div className="body-base" style={{ marginTop: '4px', fontSize: '12px' }}>
                {result.bias_flag_reason || 'No significant bias probes triggered.'}
              </div>
            </div>
          </div>
          
          {result.counterfactual_probe && result.counterfactual_probe.tested_surnames.length > 0 && (
            <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: '1px solid var(--outline-variant)' }}>
               <p className="body-base" style={{ fontSize: '12px', marginBottom: '8px', color: 'var(--outline)' }}>Tested Counterfactual Surnames:</p>
               <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                 {result.counterfactual_probe.tested_surnames.map(s => (
                   <span key={s} style={{ 
                     padding: '2px 8px', 
                     borderRadius: '4px', 
                     background: s === result.counterfactual_probe.max_delta_surname ? 'rgba(239,68,68,0.2)' : 'var(--surface-container-high)',
                     border: s === result.counterfactual_probe.max_delta_surname ? '1px solid var(--axis-dalit)' : '1px solid var(--outline-variant)',
                     fontSize: '11px'
                   }}>
                     {s}
                   </span>
                 ))}
               </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default FlagTab
