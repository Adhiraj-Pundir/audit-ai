import React, { useEffect, useState } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts'
import { AlertTriangle, Loader2 } from 'lucide-react'

const MeasureTab = () => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    fetch(`${apiUrl}/audit?mitigation=none`)
      .then(res => res.json())
      .then(json => {
        setData(json)
        setLoading(false)
      })
      .catch(err => {
        console.error("Failed to fetch audit data:", err)
        setError("Backend unavailable. Please ensure the Python server is running.")
        setLoading(false)
      })
  }, [])

  const getColor = (axis) => {
    switch (axis?.toLowerCase()) {
      case 'dalit': return 'var(--axis-dalit)'
      case 'muslim': return 'var(--axis-muslim)'
      default: return 'var(--axis-other)'
    }
  }

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const entry = payload[0].payload
      return (
        <div className="card" style={{ padding: '12px', background: 'var(--surface-container-high)', border: '1px solid var(--primary)', marginBottom: 0 }}>
          <p className="body-base" style={{ fontWeight: 600, color: 'white' }}>{entry.clean_surname} → {entry.corrupted_surname}</p>
          <p className="body-base">Delta: <span style={{ color: Math.abs(entry.delta) > 0.5 ? 'var(--axis-dalit)' : 'var(--primary)' }}>{entry.delta.toFixed(3)}</span></p>
          <p className="body-base">Axis: {entry.axis}</p>
          <p className="body-base" style={{ fontSize: '11px', opacity: 0.7 }}>Income: ₹{entry.income.toLocaleString()}</p>
        </div>
      )
    }
    return null
  }

  if (loading) return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '400px' }}>
      <Loader2 size={48} className="animate-spin" style={{ color: 'var(--primary)', marginBottom: '16px' }} />
      <p className="body-base">Loading Audit Report...</p>
    </div>
  )

  if (error) return (
    <div className="card" style={{ borderColor: 'var(--axis-dalit)', background: 'rgba(239, 68, 68, 0.05)', textAlign: 'center', padding: '48px' }}>
      <AlertTriangle size={48} style={{ color: 'var(--axis-dalit)', marginBottom: '16px' }} />
      <h3 className="headline-md">Connection Error</h3>
      <p className="body-base">{error}</p>
    </div>
  )

  return (
    <div className="fade-in">
      <div className="header-section">
        <h1 className="headline-md">Audit Analysis: Delta Distribution</h1>
        <p className="body-base">Visualizing propensity score gaps across {data.n_pairs} representative pair samples for <strong>{data.model}</strong>.</p>
      </div>

      <div className="card">
        <div className="chart-container">
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
                domain={['auto', 'auto']}
              />
              <ZAxis type="number" range={[60, 400]} />
              <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
              
              <ReferenceLine y={0.5} stroke="var(--axis-dalit)" strokeDasharray="3 3" label={{ position: 'right', value: '+0.5 Threshold', fill: 'var(--axis-dalit)', fontSize: 10 }} />
              <ReferenceLine y={-0.5} stroke="var(--axis-dalit)" strokeDasharray="3 3" label={{ position: 'right', value: '-0.5 Threshold', fill: 'var(--axis-dalit)', fontSize: 10 }} />
              <ReferenceLine y={0} stroke="var(--outline)" strokeOpacity={0.3} />

              <Scatter name="Audit Pairs" data={data.pairs}>
                {data.pairs.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={getColor(entry.axis)} 
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="legend">
          {data.by_axis.map(a => (
            <div className="legend-item" key={a.axis}>
              <div className="legend-color" style={{ background: getColor(a.axis) }}></div>
              <span style={{ textTransform: 'capitalize' }}>{a.axis} Axis (n={a.n})</span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid-summary" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '24px' }}>
         <div className="card" style={{ padding: '16px', marginBottom: 0 }}>
            <div className="body-base" style={{ fontSize: '11px', textTransform: 'uppercase' }}>Mean Delta</div>
            <div className="headline-md" style={{ margin: 0 }}>{data.summary.mean_delta.toFixed(3)}</div>
         </div>
         <div className="card" style={{ padding: '16px', marginBottom: 0 }}>
            <div className="body-base" style={{ fontSize: '11px', textTransform: 'uppercase' }}>Strong Bias Detected</div>
            <div className="headline-md" style={{ margin: 0, color: 'var(--axis-dalit)' }}>{data.summary.n_strong_bias}</div>
         </div>
         <div className="card" style={{ padding: '16px', marginBottom: 0 }}>
            <div className="body-base" style={{ fontSize: '11px', textTransform: 'uppercase' }}>Std. Deviation</div>
            <div className="headline-md" style={{ margin: 0 }}>{data.summary.std_delta.toFixed(3)}</div>
         </div>
         <div className="card" style={{ padding: '16px', marginBottom: 0 }}>
            <div className="body-base" style={{ fontSize: '11px', textTransform: 'uppercase' }}>Pairs Audited</div>
            <div className="headline-md" style={{ margin: 0 }}>{data.n_pairs}</div>
         </div>
      </div>
    </div>
  )
}

export default MeasureTab
