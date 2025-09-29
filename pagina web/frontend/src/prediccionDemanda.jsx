import { useMemo, useState } from "react";

const TEMPORADAS = [
  "Semana Santa","Navidad","Año Nuevo","Aguinaldo Boyacense",
  "Vacaciones mitad de año","Vacaciones fin de año","Puente festivo"
];
const SEGMENTOS = ["económico","precio-calidad","premium"];
const TIPOS = ["Solo","Pareja","Familia","Grupo"];

function flagsFromTemporada(t) {
  const t0 = (t || "").toLowerCase();
  return {
    es_evento_ciudad: t0 === "aguinaldo boyacense" ? 1 : 0,
    es_semana_santa:  t0 === "semana santa" ? 1 : 0,
    es_navidad:       t0 === "navidad" ? 1 : 0,
    es_puente:        t0.startsWith("puente festivo") ? 1 : 0,
  };
}
function defaultsByTemporada(t) {
  switch (t) {
    case "Semana Santa": return { duracion_dias: 8,  mes_inicio: 4 };
    case "Navidad": return { duracion_dias: 5,  mes_inicio: 12 };
    case "Año Nuevo": return { duracion_dias: 4,  mes_inicio: 1 };
    case "Aguinaldo Boyacense": return { duracion_dias: 7,  mes_inicio: 12 };
    case "Vacaciones mitad de año": return { duracion_dias: 31, mes_inicio: 6 };
    case "Vacaciones fin de año":  return { duracion_dias: 15, mes_inicio: 12 };
    case "Puente festivo":         return { duracion_dias: 3,  mes_inicio: 6 };
    default: return { duracion_dias: 5, mes_inicio: 1 };
  }
}

export default function PrediccionDemanda() {
  const [anio, setAnio] = useState(2025);
  const [temporada, setTemporada] = useState("Semana Santa");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const apiBase = useMemo(() => "http://127.0.0.1:8000", []);

  function buildBatch(t, y) {
    const base = { anio: Number(y), ...defaultsByTemporada(t), ...flagsFromTemporada(t) };
    const items = [];
    for (const segmento of SEGMENTOS) {
      for (const Tipo_viajero of TIPOS) {
        items.push({
          temporada: t, segmento, Tipo_viajero,
          anio: base.anio,
          duracion_dias: base.duracion_dias,
          mes_inicio: base.mes_inicio,
          es_evento_ciudad: base.es_evento_ciudad,
          es_semana_santa:  base.es_semana_santa,
          es_navidad:       base.es_navidad,
          es_puente:        base.es_puente,
        });
      }
    }
    return items;
  }

  async function predict() {
    setLoading(true);
    setResult(null);
    try {
      const items = buildBatch(temporada, anio);     // ← 12 combinaciones auto
      const res = await fetch(`${apiBase}/predict_demand`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items })
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      alert("Error al predecir: " + e);
    } finally {
      setLoading(false);
    }
  }

  const detail = result?.detail?.[temporada] || null;   // solo la temporada elegida
  const totalTemporada = result?.totals?.by_temporada?.[temporada];

  return (
    <div className="p-4 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold mb-3">Predecir demanda hotelera</h1>

      {/* Controles mínimos */}
      <div className="flex flex-wrap gap-3 items-end bg-gray-50 p-4 rounded-xl shadow">
        <label className="block">
          <div className="text-sm">Temporada</div>
          <select
            value={temporada}
            onChange={(e)=>setTemporada(e.target.value)}
            className="w-60 p-2 rounded border"
          >
            {TEMPORADAS.map(x => <option key={x}>{x}</option>)}
          </select>
        </label>
        <label className="block">
          <div className="text-sm">Año</div>
          <input
            type="number"
            value={anio}
            onChange={(e)=>setAnio(e.target.value)}
            className="w-32 p-2 rounded border"
          />
        </label>
        <button
          onClick={predict}
          disabled={loading}
          className="ml-auto px-4 py-2 rounded bg-indigo-600 text-white"
        >
          {loading ? "Calculando..." : "Predecir demanda"}
        </button>

        {totalTemporada!=null && (
          <div className="text-sm text-gray-700">
            Total {temporada} {anio}: <b>{Number(totalTemporada).toLocaleString()}</b> personas
          </div>
        )}
      </div>

      {/* Resultados */}
      {result && (
        <div className="mt-6 text-left">
          <h2 className="text-xl font-semibold mb-3">Resultados</h2>

          {/* Totales rápidos */}
          <div className="grid md:grid-cols-3 gap-3">
            <Card title="Total por temporada" data={result.totals?.by_temporada}/>
            <Card title="Total por segmento" data={result.totals?.by_segmento}/>
            <Card title="Total por tipo" data={result.totals?.by_tipo}/>
          </div>

          {/* Detalle: temporada → segmento → tipos */}
          {detail && (
            <div className="mt-6">
              <h3 className="text-lg font-bold">{temporada} · {anio}</h3>
              <div className="grid md:grid-cols-3 gap-3 mt-2">
                {Object.entries(detail).map(([seg, rows])=>(
                  <div key={seg} className="border rounded-lg p-3 bg-white shadow-sm">
                    <div className="font-semibold mb-2">{seg}</div>
                    <table className="w-full text-sm">
                      <thead><tr><th className="text-left">Tipo</th><th className="text-right">Personas</th></tr></thead>
                      <tbody>
                        {rows.map((r,i)=>(
                          <tr key={i} className="border-t">
                            <td>{r.Tipo_viajero}</td>
                            <td className="text-right">{Number(r.personas_pred).toLocaleString()}</td>
                          </tr>
                        ))}
                        <tr className="border-t font-semibold">
                          <td>Total segmento</td>
                          <td className="text-right">
                            {rows.reduce((s,x)=>s+Number(x.personas_pred||0),0).toLocaleString()}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Reutiliza tu Card actual tal cual
function Card({title, data}) {
  const entries = Object.entries(data || {});
  return (
    <div className="border rounded-lg p-3 bg-white shadow-sm">
      <div className="font-semibold mb-2">{title}</div>
      {!entries.length ? <div className="text-sm text-gray-500">—</div> : (
        <ul className="text-sm">
          {entries.map(([k,v])=>(
            <li key={k} className="flex justify-between border-t py-1">
              <span>{k}</span><span>{Number(v).toLocaleString()}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
