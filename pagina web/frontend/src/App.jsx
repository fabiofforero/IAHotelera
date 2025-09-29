


import React, { useEffect, useState } from "react";
import PrediccionDemanda from "./PrediccionDemanda";

// === Ajusta esto a tu backend ===
const API_URL = "http://127.0.0.1:8000"; // FastAPI

// Utilidades GET/POST
async function apiGet(path) {
  const r = await fetch(`${API_URL}${path}`);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}
async function apiPost(path, payload) {
  const r = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    let msg = `POST ${path} → ${r.status}`;
    try {
      const j = await r.json();
      if (j?.detail) msg += ` · ${j.detail}`;
    } catch {}
    throw new Error(msg);
  }
  return r.json();
}

// Tarjeta de hotel
function HotelCard({ h }) {
  return (
    <div
      className="rounded-2xl shadow p-4 bg-white/70 backdrop-blur border border-gray-200"
      aria-label={`Ficha de hotel ${h.nombre}`}
    >
      <div className="text-lg font-semibold">{h.nombre}</div>
      <div className="text-sm text-gray-600 capitalize">{h.segmento}</div>

      {(h.direccion || h.telefono || h.sitio_web) && (
        <div className="mt-2 text-sm">
          {h.direccion && (
            <div>
              <span className="font-medium">Dirección:</span> {h.direccion}
            </div>
          )}
          {h.telefono && (
            <div>
              <span className="font-medium">Tel:</span> {h.telefono}
            </div>
          )}
          {h.sitio_web && (
            <div>
              <a
                className="text-blue-600 underline"
                href={h.sitio_web}
                target="_blank"
                rel="noreferrer"
                aria-label="Abrir sitio del hotel"
              >
                Sitio web
              </a>
            </div>
          )}
        </div>
      )}

      {(h.precio_promedio || h.rating) && (
        <div className="mt-2 text-sm flex gap-4">
          {h.precio_promedio && <span>Precio: {h.precio_promedio}</span>}
          {h.rating && <span>⭐ {h.rating}</span>}
        </div>
      )}
    </div>
  );
}

// Lista por segmento (botón carga 5 hoteles)
function SegmentList({ segmento }) {
  const [hoteles, setHoteles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function load() {
    try {
      setLoading(true);
      setError("");
      const data = await apiGet(
        `/hotels?segment=${encodeURIComponent(segmento)}&limit=5`
      );
      setHoteles(data.hoteles || []);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  // 🚀 Auto-carga al montar y cuando cambie el segmento
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segmento]);

  return (
    <section className="mt-6" aria-label={`Hoteles del segmento ${segmento}`}>
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold capitalize">{segmento}</h2>
        {/* Botón opcional por si quieres recargar manualmente */}
        <button
          onClick={load}
          className="px-4 py-2 rounded-xl bg-gray-900 text-white hover:bg-black active:scale-[.99]"
        >
          Refrescar
        </button>
      </div>
      {loading && <p className="mt-4">Cargando…</p>}
      {error && <p className="mt-4 text-red-600">{error}</p>}
      <div className="mt-4 grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {hoteles.map((h, i) => (
          <HotelCard key={i} h={h} />
        ))}
      </div>
    </section>
  );
}

const opciones = {
  genero: [
    { label: "Femenino", value: 0 },
    { label: "Masculino", value: 1 },
  ],
  bool: [
    { label: "No", value: 0 },
    { label: "Sí", value: 1 },
  ],
  tipo_viajero: ["Familia", "Grupo", "Individual", "Pareja"],
  motivo_viaje: ["Turismo", "Trabajo", "Eventos culturales"],
  temporada: ["Vacaciones", "Semana Santa", "Puentes festivos"],
  canal: ["Directo", "Booking", "Airbnb", "Agencia física"],
  ocupacion: ["Alta", "Media", "Baja"],
  reputacion_online: ["Alta", "Media", "Baja"],
  limpieza: ["Regular", "Buena", "Excelente"],
  procedencia: ["Bogotá", "EEUU", "España", "Medellín", "México", "Tunja"],
};

function PreferenciasSugerencias({ segmento }) {
  const [hoteles, setHoteles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function load() {
    try {
      setLoading(true);
      setError("");
      const data = await apiGet(
        `/hotels?segment=${encodeURIComponent(segmento)}&limit=5`
      );
      setHoteles(data.hoteles || []);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  // Auto-carga cuando cambia el segmento
  useEffect(() => {
    if (segmento) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segmento]);

  if (!segmento) return null;

  return (
    <div id="sugerencias" className="mt-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">
          Sugerencias para “{segmento}”
        </h3>
        <button
          onClick={load}
          className="px-4 py-2 rounded-xl bg-gray-900 text-white hover:bg-black"
        >
          Recargar
        </button>
      </div>
      {loading && <p className="mt-3">Cargando…</p>}
      {error && <p className="mt-3 text-red-600">{error}</p>}
      <div className="mt-4 grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {hoteles.map((h, i) => (
          <HotelCard key={i} h={h} />
        ))}
      </div>
    </div>
  );
}

function PreferenciasForm() {
  const [form, setForm] = useState({
    Edad: 35,
    Género: 1,
    Duración_estadía: 3,
    Anticipación_reserva_días: 7,
    Calificación: 3.8,
    WiFi: 1,
    Parqueadero: 0,
    Spa_gimnasio: 1,
    Restaurante: 1,
    Desayuno_incluido: 1,
    Sostenibilidad: 0,
    PetFriendly: 1,
    Fidelización: 0,
    Tipo_viajero: "Pareja",
    Motivo_viaje: "Turismo",
    Temporada: "Navidad",
    Canal_reserva: "Booking",
    Ocupación: "Media",
    Reputación_online: "Media",
    Limpieza: "Buena",
    Procedencia: "Bogotá",
    Precio: 60000, // ← NUEVO (opcional). Si lo dejas vacío, no se usa el ajuste por precio
  });
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const segmento = resp?.segmento || null;

  async function onSubmit(e) {
    e.preventDefault();
    try {
      setLoading(true);
      setError("");
      setResp(null);

      const payload = {
        Edad: Number(form.Edad),
        Género: Number(form.Género),
        Duración_estadía: Number(form.Duración_estadía),
        Anticipación_reserva_días: Number(form.Anticipación_reserva_días),
        Calificación: Number(form.Calificación),
        WiFi: Number(form.WiFi),
        Parqueadero: Number(form.Parqueadero),
        Spa_gimnasio: Number(form.Spa_gimnasio),
        Restaurante: Number(form.Restaurante),
        Desayuno_incluido: Number(form.Desayuno_incluido),
        Sostenibilidad: Number(form.Sostenibilidad),
        PetFriendly: Number(form.PetFriendly),
        Fidelización: Number(form.Fidelización),
        Tipo_viajero: form.Tipo_viajero,
        Motivo_viaje: form.Motivo_viaje,
        Temporada: form.Temporada,
        Canal_reserva: form.Canal_reserva,
        Ocupación: form.Ocupación,
        Reputación_online: form.Reputación_online,
        Limpieza: form.Limpieza,
        Procedencia: form.Procedencia,
      };

      // Solo enviar Precio si es numérico
      const precioN = Number(form.Precio);
      if (!Number.isNaN(precioN) && String(form.Precio).trim() !== "") {
        payload.Precio = precioN;
      }

      const data = await apiPost("/predict", payload);
      setResp(data);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="mt-6" aria-label="Recomiéndame por preferencias">
      <h2 className="text-xl font-bold">Recomiéndame por preferencias</h2>
      <form onSubmit={onSubmit} className="mt-4 grid md:grid-cols-2 gap-4">
        {/* numéricos */}
        <div className="p-4 rounded-2xl border bg-white/70">
          <label className="block text-sm font-medium">Edad</label>
          <input
            type="number"
            min={18}
            max={95}
            value={form.Edad}
            onChange={(e) => setForm({ ...form, Edad: e.target.value })}
            className="mt-1 w-full border rounded p-2"
          />
          <label className="block text-sm font-medium mt-3">
            Duración de estadía (días)
          </label>
          <input
            type="number"
            min={1}
            max={60}
            value={form.Duración_estadía}
            onChange={(e) =>
              setForm({ ...form, Duración_estadía: e.target.value })
            }
            className="mt-1 w-full border rounded p-2"
          />
          <label className="block text-sm font-medium mt-3">
            Anticipación de reserva (días)
          </label>
          <input
            type="number"
            min={0}
            max={365}
            value={form.Anticipación_reserva_días}
            onChange={(e) =>
              setForm({ ...form, Anticipación_reserva_días: e.target.value })
            }
            className="mt-1 w-full border rounded p-2"
          />
          <label className="block text-sm font-medium mt-3">
            Calificación esperada (0–5)
          </label>
          <input
            type="number"
            step="0.1"
            min={0}
            max={5}
            value={form.Calificación}
            onChange={(e) =>
              setForm({ ...form, Calificación: e.target.value })
            }
            className="mt-1 w-full border rounded p-2"
          />

          {/* NUEVO: Precio objetivo */}
          <label className="block text-sm font-medium mt-3">
            Precio objetivo (opcional)
          </label>
          <input
            type="number"
            min={0}
            placeholder="Ej: 150000"
            value={form.Precio}
            onChange={(e) => setForm({ ...form, Precio: e.target.value })}
            className="mt-1 w-full border rounded p-2"
          />
        </div>

        {/* booleanos y categóricos */}
        <div className="p-4 rounded-2xl border bg-white/70 grid grid-cols-2 gap-3">
          <label className="text-sm">Género</label>
          <select
            value={form.Género}
            onChange={(e) => setForm({ ...form, Género: e.target.value })}
            className="border rounded p-2"
          >
            {opciones.genero.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>

          {[
            "WiFi",
            "Parqueadero",
            "Gimnasio",
            "Restaurante",
            "Desayuno_incluido",
            "Sostenibilidad y ecológico",
            "PetFriendly",
            "Fidelización",
          ].map((k) => (
            <React.Fragment key={k}>
              <label className="text-sm">{k.replace(/_/g, " ")}</label>
              <select
                value={form[k]}
                onChange={(e) => setForm({ ...form, [k]: e.target.value })}
                className="border rounded p-2"
              >
                {opciones.bool.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </React.Fragment>
          ))}

          <label className="text-sm">Tipo de viajero</label>
          <select
            value={form.Tipo_viajero}
            onChange={(e) => setForm({ ...form, Tipo_viajero: e.target.value })}
            className="border rounded p-2"
          >
            {opciones.tipo_viajero.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>

          <label className="text-sm">Motivo del viaje</label>
          <select
            value={form.Motivo_viaje}
            onChange={(e) => setForm({ ...form, Motivo_viaje: e.target.value })}
            className="border rounded p-2"
          >
            {opciones.motivo_viaje.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>

          <label className="text-sm">Temporada</label>
          <select
            value={form.Temporada}
            onChange={(e) => setForm({ ...form, Temporada: e.target.value })}
            className="border rounded p-2"
          >
            {opciones.temporada.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>

          <label className="text-sm">Canal de reserva</label>
          <select
            value={form.Canal_reserva}
            onChange={(e) => setForm({ ...form, Canal_reserva: e.target.value })}
            className="border rounded p-2"
          >
            {opciones.canal.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>

          <label className="text-sm">Ocupación</label>
          <select
            value={form.Ocupación}
            onChange={(e) => setForm({ ...form, Ocupación: e.target.value })}
            className="border rounded p-2"
          >
            {opciones.ocupacion.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>

          <label className="text-sm">Reputación online</label>
          <select
            value={form.Reputación_online}
            onChange={(e) =>
              setForm({ ...form, Reputación_online: e.target.value })
            }
            className="border rounded p-2"
          >
            {opciones.reputacion_online.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>

          <label className="text-sm">Limpieza</label>
          <select
            value={form.Limpieza}
            onChange={(e) => setForm({ ...form, Limpieza: e.target.value })}
            className="border rounded p-2"
          >
            {opciones.limpieza.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>

          <label className="text-sm">Procedencia</label>
          <select
            value={form.Procedencia}
            onChange={(e) => setForm({ ...form, Procedencia: e.target.value })}
            className="border rounded p-2"
          >
            {opciones.procedencia.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>

        <div className="md:col-span-2 flex gap-3">
          <button
            type="submit"
            className="px-5 py-2 rounded-xl bg-emerald-600 text-white hover:bg-emerald-700"
          >
            Predecir segmento
          </button>
        </div>
      </form>

      {loading && <p className="mt-4">Procesando…</p>}
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {resp && (
        <div className="mt-6 p-4 rounded-2xl border bg-white/70">
          <p className="text-lg">
            Segmento predicho:{" "}
            <span className="font-semibold capitalize">{resp.segmento}</span>
            {resp.ajuste_precio ? " · (ajustado por precio)" : ""}
          </p>
          <p className="text-sm text-gray-600">
            Probabilidades:{" "}
            {Object.entries(resp.probabilidades)
              .map(([k, v]) => `${k}: ${Number(v).toFixed(3)}`)
              .join(" · ")}
          </p>

          {/* Carga automática de 5 sugerencias para el segmento predicho */}
          <PreferenciasSugerencias segmento={resp.segmento} />
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [vista, setVista] = useState("home");

  const botones = [
    { id: "economico", label: "Hoteles económicos", segmento: "económico" },
    {
      id: "precio",
      label: "Hoteles precio-calidad",
      segmento: "precio-calidad",
    },
    { id: "premium", label: "Hoteles premium", segmento: "premium" },
    { id: "preferencias", label: "Recomiéndame por preferencias" },
    { id: "demanda", label: "Predecir demanda hotelera" },

    
  ];

  return (
    <main className="min-h-screen bg-gradient-to-b from-sky-50 to-emerald-50 text-gray-900">
      <header className="px-6 py-5 border-b bg-white/70 backdrop-blur">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold">Hoteles en Tunja</h1>
          <nav className="flex gap-2">
            {botones.map((b) => (
              <button
                key={b.id}
                onClick={() => setVista(b.id)}
                className={`px-4 py-2 rounded-xl border ${
                  vista === b.id
                    ? "bg-gray-900 text-white"
                    : "bg-white hover:bg-gray-50"
                }`}
              >
                {b.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {vista === "home" && (
          <section className="text-center" aria-label="Inicio">
            <h2 className="text-3xl font-bold">Elige una sección</h2>
            <p className="mt-2 text-gray-600">
              Explora por segmento o recibe recomendaciones personalizadas.
            </p>
            <div className="mt-6 grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              {botones.map((b) => (
                <button
                  key={b.id}
                  onClick={() => setVista(b.id)}
                  className="p-6 rounded-2xl border bg-white/80 hover:bg-white active:scale-[.99]"
                >
                  <div className="text-lg font-semibold">{b.label}</div>
                </button>
              ))}
            </div>
          </section>
        )}

        {vista === "economico" && <SegmentList segmento="económico" />}
        {vista === "precio" && <SegmentList segmento="precio-calidad" />}
        {vista === "premium" && <SegmentList segmento="premium" />}
        {vista === "preferencias" && <PreferenciasForm />}
        {vista === "demanda" && <PrediccionDemanda />}

      </div>

      <footer className="px-6 py-10 mt-10 border-t text-center text-sm text-gray-500">
        <span id="year" aria-label="Año actual">
          {new Date().getFullYear()}
        </span>{" "}
        • Prototipo académico · Universidad de Boyacá
      </footer>
    </main>
  );
}
