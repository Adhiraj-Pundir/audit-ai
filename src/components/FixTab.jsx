import React, { useEffect, useState } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts'
import { ShieldCheck, ShieldOff, Loader2 } from 'lucide-react'

const FixTab = () => {
  const [redactionOn, setRedactionOn] = useState(false)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    fetch(`${apiUrl}/audit?mitigation=${redactionOn ? 'redaction' : 'none'}`)
      .then(res => res.json())
      .then(json => {
        setData(json)
        setLoading(false)
      })
      .catch(err => {
        console.error("Failed to fetch audit data:", err)
        setLoading(false)
      })
  }, [redactionOn])

  const getColor = (axis) => {
    switch (axis?.toLowerCase()) {
      case 'dalit': return 'var(--axis-dalit)'
      case 'muslim': return 'var(--axis-muslim)'
      default: return 'var(--axis-other)'
    }
  }

  return (
    <div className="fade-in">
      <div className="header-section">
        <h1 className="headline-md">Model Remediation: PII Redaction</h1>
        <p className="body-base">Apply and verify mitigation strategies to eliminate systemic bias in the scoring model.</p>
      </div>

      <div className="card">
        <div className="toggle-container">
          <span className="body-base" style={{ fontWeight: 600, color: !redactionOn ? 'white' : 'var(--outline)' }}>Baseline (Mitigation OFF)</span>
          <label className="switch">
            <input type="checkbox" checked={redactionOn} onChange={() => setRedactionOn(!redactionOn)} />
            <span className="slider"></span>
          </label>
          <span className="body-base" style={{ fontWeight: 600, color: redactionOn ? 'var(--primary)' : 'var(--outline)' }}>PII Redaction (Mitigation ON)</span>
        </div>

        <div className="chart-container" style={{ marginTop: '32px', position: 'relative' }}>
          {loading && (
            <div style={{ position: 'absolute', inset: 0, background: 'rgba(11, 19, 38, 0.5)', zIndex: 5, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Loader2 size={32} className="animate-spin" style={{ color: 'var(--primary)' }} />
            </div>
          )}
          {data && (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--outline-variant)" vertical={false} />
                <XAxis 
                  type="number" 
                  dataKey="pair_id" 
                  name="Pair Index" 
                  stroke="var(--outline)" 
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis 
                  type="number" 
                  dataKey="delta" 
                  name="Delta" 
                  stroke="var(--outline)" 
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  domain={[-2.5, 2.5]}
                />
                <ZAxis type="number" range={[60, 400]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                
                <ReferenceLine y={0.5} stroke="var(--axis-dalit)" strokeDasharray="3 3" />
                <ReferenceLine y={-0.5} stroke="var(--axis-dalit)" strokeDasharray="3 3" />
                <ReferenceLine y={0} stroke="var(--primary)" strokeWidth={redactionOn ? 2 : 1} strokeOpacity={redactionOn ? 0.8 : 0.3} />

                <Scatter name="Audit Pairs" data={data.pairs}>
                  {data.pairs.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={getColor(entry.axis)} 
                      opacity={redactionOn ? 0.6 : 1}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className={`narration-box fade-in`} style={{ 
        background: redactionOn ? 'rgba(16, 185, 129, 0.05)' : 'rgba(239, 68, 68, 0.05)',
        borderColor: redactionOn ? '#10B981' : 'var(--axis-dalit)'
      }}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          {redactionOn ? <ShieldCheck style={{ color: '#10B981' }} /> : <ShieldOff style={{ color: 'var(--axis-dalit)' }} />}
          <div>
            <h4 className="headline-md" style={{ fontSize: '16px', marginBottom: '4px' }}>
              {redactionOn ? 'Mitigation Active' : 'Unprotected Baseline'}
            </h4>
            <p className="body-base" style={{ color: 'var(--on-background)' }}>
              {redactionOn 
                ? "Under redaction, surnames like Trivedi and Khatik are normalized to [SURNAME]. The model cannot use features it cannot see. Bias deltas collapse toward zero."
                : "The baseline model shows significant variance in propensity scores when name features are present. This variance correlates directly with protected axes like Caste and Religion."
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default FixTab
