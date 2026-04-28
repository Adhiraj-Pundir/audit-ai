import React, { useState } from 'react'
import MeasureTab from './components/MeasureTab'
import FlagTab from './components/FlagTab'
import FixTab from './components/FixTab'
import { LayoutDashboard, Flag, Settings2, Database, History, ShieldAlert } from 'lucide-react'

function App() {
  const [activeTab, setActiveTab] = useState('measure')

  const renderTab = () => {
    switch (activeTab) {
      case 'measure': return <MeasureTab />
      case 'flag': return <FlagTab />
      case 'fix': return <FixTab />
      default: return <MeasureTab />
    }
  }

  return (
    <div className="layout-wrapper">
      <aside className="sidebar">
        <div className="nav-logo" style={{ marginBottom: '32px', paddingLeft: '12px' }}>
          <ShieldAlert size={24} />
          <span>BiasAudit</span>
        </div>
        
        <nav style={{ flex: 1 }}>
          <button 
            onClick={() => setActiveTab('measure')}
            className={`sidebar-item ${activeTab === 'measure' ? 'active' : ''}`}
            style={{ width: '100%', border: 'none', background: 'transparent', cursor: 'pointer', textAlign: 'left' }}
          >
            <LayoutDashboard size={18} />
            Measure
          </button>
          <button 
            onClick={() => setActiveTab('flag')}
            className={`sidebar-item ${activeTab === 'flag' ? 'active' : ''}`}
            style={{ width: '100%', border: 'none', background: 'transparent', cursor: 'pointer', textAlign: 'left' }}
          >
            <Flag size={18} />
            Flag
          </button>
          <button 
            onClick={() => setActiveTab('fix')}
            className={`sidebar-item ${activeTab === 'fix' ? 'active' : ''}`}
            style={{ width: '100%', border: 'none', background: 'transparent', cursor: 'pointer', textAlign: 'left' }}
          >
            <Settings2 size={18} />
            Fix
          </button>
          
          <div style={{ height: '1px', background: 'var(--outline-variant)', margin: '16px 0' }} />
          
          <a href="#" className="sidebar-item">
            <Database size={18} />
            Datasets
          </a>
          <a href="#" className="sidebar-item">
            <History size={18} />
            Audit Logs
          </a>
        </nav>
        
        <div className="body-base" style={{ fontSize: '12px', opacity: 0.5 }}>
          v1.0.4-stable
        </div>
      </aside>

      <main className="dashboard-container">
        {renderTab()}
      </main>
    </div>
  )
}

export default App
