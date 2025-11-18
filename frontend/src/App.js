import { useCallback, useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import Select from 'react-select';
import './App.css';

const NUTS_VERSIONS = ['2006', '2010', '2013', '2016', '2021'];
const MONTHS = [
  { value: '01', label: 'January' },
  { value: '02', label: 'February' },
  { value: '03', label: 'March' },
  { value: '04', label: 'April' },
  { value: '05', label: 'May' },
  { value: '06', label: 'June' },
  { value: '07', label: 'July' },
  { value: '08', label: 'August' },
  { value: '09', label: 'September' },
  { value: '10', label: 'October' },
  { value: '11', label: 'November' },
  { value: '12', label: 'December' },
];

const VERSION_YEAR_RANGES = {
  '2006': { min: 2008, max: 2011 },
  '2010': { min: 2012, max: 2014 },
  '2013': { min: 2015, max: 2017 },
  '2016': { min: 2018, max: 2020 },
  '2021': { min: 2021, max: 2021 },
};

function App() {
  const [selectedVersion, setSelectedVersion] = useState('2021');
  const [regions, setRegions] = useState([]);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [year, setYear] = useState(2021);
  const [month, setMonth] = useState(MONTHS[0]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [minYear, setMinYear] = useState(2021);
  const [maxYear, setMaxYear] = useState(2021);

  const geoData = useRef(null);
  const geoJsonLayer = useRef(null);
  const mapRef = useRef(null);

  const loadGeoJSON = useCallback(async (version) => {
    try {
      setError(null);
      const response = await fetch(`/nuts3_${version}.geojson`);
      console.log('Fetch response:', response.status, response.statusText, response.url);
      if (!response.ok) throw new Error(`GeoJSON not found for version ${version}`);
      let data = await response.json();

      data.features = data.features.filter((f) => f.properties.LEVL_CODE === 3);
      geoData.current = data;

      if (geoJsonLayer.current) geoJsonLayer.current.remove();

      geoJsonLayer.current = L.geoJSON(data, {
        style: { color: '#555', weight: 1, fillOpacity: 0.1 },
      }).addTo(mapRef.current._leaflet_map);

      const regionOptions = data.features.map((f) => ({
        value: f.properties.NUTS_ID,
        label: `${f.properties.NUTS_ID} — ${f.properties.NAME_LATN}`,
      }));

      setRegions(regionOptions);
    } catch (err) {
      console.error('Failed to load GeoJSON for version', version, err);
      setError(err.message || 'Failed to load GeoJSON');
    }
  }, []);

  useEffect(() => {
    const map = L.map(mapRef.current, {
      center: [54, 15],
      zoom: 3.3,
      dragging: false,
      zoomControl: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
      boxZoom: false,
      keyboard: false,
      tap: false,
      touchZoom: false,
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);

    mapRef.current._leaflet_map = map;
    loadGeoJSON(selectedVersion);

    return () => map.remove();
  }, [loadGeoJSON, selectedVersion]);

  const highlightRegion = useCallback((regionOption, prob) => {
    if (!geoData.current || !regionOption) return;
    const map = mapRef.current._leaflet_map;

    if (geoJsonLayer.current) {
      geoJsonLayer.current.clearLayers();
      geoJsonLayer.current.addData(geoData.current);
    }

    const feature = geoData.current.features.find(
      (f) => f.properties.NUTS_ID === regionOption.value
    );

    if (feature) {
      // Calculate color based on probability: 0 blue, 0.63 white, 1 orange
      let fillColor = 'orange'; // default
      if (prob != null && !isNaN(prob)) {
        const p = Math.max(0, Math.min(1, prob)); // clamp 0-1
        const threshold = 0.63;
        if (p <= threshold) {
          const t = p / threshold;
          const saturation = 100 * (1 - t);
          const lightness = 50 + 50 * t;
          fillColor = `hsl(240, ${saturation}%, ${lightness}%)`;
        } else {
          const t = (p - threshold) / (1 - threshold);
          const saturation = 100 * t;
          const lightness = 100 - 50 * t;
          fillColor = `hsl(30, ${saturation}%, ${lightness}%)`;
        }
      }

      const layer = L.geoJSON(feature, {
        style: { color: fillColor, weight: 2, fillColor, fillOpacity: 0.3 },
      }).addTo(map);
      map.fitBounds(layer.getBounds().pad(0.2), { animate: false });
    }
  }, []);

  const handleFetchData = useCallback(async () => {
    if (!selectedRegion) return;

    const yearMonth = `${year}-${month.value}`;

    try {
      setLoading(true);
      setError(null);

      const res = await fetch('http://localhost:8000/lepto/precipitation/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          version: selectedVersion,
          region_id: selectedRegion.value,
          year_month: yearMonth,
        }),
      });

      let payload;
      try {
        payload = await res.json();
      } catch (jsonError) {
        throw new Error('Server returned invalid response. Ensure backend is running.');
      }

      if (!res.ok) {
        throw new Error(payload?.error || 'Request failed');
      }

      setData(payload);
      highlightRegion(selectedRegion, payload.predicted_probability);
    } catch (err) {
      console.error('Failed to fetch data:', err);
      setError(err.message || 'Failed to fetch');
    } finally {
      setLoading(false);
    }
  }, [selectedRegion, year, month, selectedVersion, highlightRegion]);

  const handleVersionChange = useCallback((e) => {
    const v = e.target.value;
    setSelectedVersion(v);
    const range = VERSION_YEAR_RANGES[v];
    setMinYear(range.min);
    setMaxYear(range.max);
    if (year < range.min || year > range.max) {
      setYear(range.min);
    }
    loadGeoJSON(v);
  }, [loadGeoJSON, year]);

  const yearOptions = Array.from({ length: maxYear - minYear + 1 }, (_, i) => minYear + i);

  // Commented out features array and related code
  /*
  const landCoverClasses = { ...DEFAULT_LAND_COVER_CLASSES, ...(data?.land_cover_percentages?.classes || {}) };

  const features = [
    {
      title: '🌧️ Precipitation Monthly Mean',
      value: data?.precipitation?.monthly_mean != null && !isNaN(data?.precipitation?.monthly_mean) ? Number(data?.precipitation?.monthly_mean).toFixed(5) : '-',
      units: data?.precipitation?.units,
    },
    {
      title: '🌧️ Precipitation Min Hourly',
      value: data?.precipitation?.min_hourly_mean != null && !isNaN(data?.precipitation?.min_hourly_mean) ? Number(data?.precipitation?.min_hourly_mean).toFixed(5) : '-',
      units: data?.precipitation?.units,
    },
    {
      title: '🌧️ Precipitation Max Hourly',
      value: data?.precipitation?.max_hourly_mean != null && !isNaN(data?.precipitation?.max_hourly_mean) ? Number(data?.precipitation?.max_hourly_mean).toFixed(5) : '-',
      units: data?.precipitation?.units,
    },
    {
      title: '🌧️ Precipitation Range Hourly',
      value: data?.precipitation?.range_hourly_mean != null && !isNaN(data?.precipitation?.range_hourly_mean) ? Number(data?.precipitation?.range_hourly_mean).toFixed(5) : '-',
      units: data?.precipitation?.units,
    },
    {
      title: '🌧️ Previous Month Precipitation Monthly Mean',
      value: data?.previous_precipitation?.monthly_mean != null && !isNaN(data?.previous_precipitation?.monthly_mean) ? Number(data?.previous_precipitation?.monthly_mean).toFixed(5) : '-',
      units: data?.previous_precipitation?.units,
    },
    {
      title: '🌧️ Previous Month Precipitation Min Hourly',
      value: data?.previous_precipitation?.min_hourly_mean != null && !isNaN(data?.previous_precipitation?.min_hourly_mean) ? Number(data?.previous_precipitation?.min_hourly_mean).toFixed(5) : '-',
      units: data?.previous_precipitation?.units,
    },
    {
      title: '🌧️ Previous Month Precipitation Max Hourly',
      value: data?.previous_precipitation?.max_hourly_mean != null && !isNaN(data?.previous_precipitation?.max_hourly_mean) ? Number(data?.previous_precipitation?.max_hourly_mean).toFixed(5) : '-',
      units: data?.previous_precipitation?.units,
    },
    {
      title: '🌧️ Previous Month Precipitation Range Hourly',
      value: data?.previous_precipitation?.range_hourly_mean != null && !isNaN(data?.previous_precipitation?.range_hourly_mean) ? Number(data?.previous_precipitation?.range_hourly_mean).toFixed(5) : '-',
      units: data?.previous_precipitation?.units,
    },
    {
      title: '🌧️ Three Months Prior Precipitation Monthly Mean',
      value: data?.three_months_prior_precipitation?.monthly_mean != null && !isNaN(data?.three_months_prior_precipitation?.monthly_mean) ? Number(data?.three_months_prior_precipitation?.monthly_mean).toFixed(5) : '-',
      units: data?.three_months_prior_precipitation?.units,
    },
    {
      title: '🌧️ Three Months Prior Precipitation Min Hourly',
      value: data?.three_months_prior_precipitation?.min_hourly_mean != null && !isNaN(data?.three_months_prior_precipitation?.min_hourly_mean) ? Number(data?.three_months_prior_precipitation?.min_hourly_mean).toFixed(5) : '-',
      units: data?.three_months_prior_precipitation?.units,
    },
    {
      title: '🌧️ Three Months Prior Precipitation Max Hourly',
      value: data?.three_months_prior_precipitation?.max_hourly_mean != null && !isNaN(data?.three_months_prior_precipitation?.max_hourly_mean) ? Number(data?.three_months_prior_precipitation?.max_hourly_mean).toFixed(5) : '-',
      units: data?.three_months_prior_precipitation?.units,
    },
    {
      title: '🌧️ Three Months Prior Precipitation Range Hourly',
      value: data?.three_months_prior_precipitation?.range_hourly_mean != null && !isNaN(data?.three_months_prior_precipitation?.range_hourly_mean) ? Number(data?.three_months_prior_precipitation?.range_hourly_mean).toFixed(5) : '-',
      units: data?.three_months_prior_precipitation?.units,
    },
    {
      title: '🌡️ Air Temperature Monthly Mean',
      value: data?.temperature?.monthly_mean != null && !isNaN(data?.temperature?.monthly_mean) ? Number(data?.temperature?.monthly_mean).toFixed(5) : '-',
      units: data?.temperature?.units,
    },
    {
      title: '🌡️ Air Temperature Min Hourly',
      value: data?.temperature?.min_hourly_mean != null && !isNaN(data?.temperature?.min_hourly_mean) ? Number(data?.temperature?.min_hourly_mean).toFixed(5) : '-',
      units: data?.temperature?.units,
    },
    {
      title: '🌡️ Air Temperature Max Hourly',
      value: data?.temperature?.max_hourly_mean != null && !isNaN(data?.temperature?.max_hourly_mean) ? Number(data?.temperature?.max_hourly_mean).toFixed(5) : '-',
      units: data?.temperature?.units,
    },
    {
      title: '🌡️ Air Temperature Range Hourly',
      value: data?.temperature?.range_hourly_mean != null && !isNaN(data?.temperature?.range_hourly_mean) ? Number(data?.temperature?.range_hourly_mean).toFixed(5) : '-',
      units: data?.temperature?.units,
    },
    {
      title: '🌱 Soil Moisture Monthly Mean',
      value: data?.soil_moisture?.monthly_mean != null && !isNaN(data?.soil_moisture?.monthly_mean) ? Number(data?.soil_moisture?.monthly_mean).toFixed(5) : '-',
      units: data?.soil_moisture?.units,
    },
    {
      title: '🌱 Soil Moisture Min Hourly',
      value: data?.soil_moisture?.min_hourly_mean != null && !isNaN(data?.soil_moisture?.min_hourly_mean) ? Number(data?.soil_moisture?.min_hourly_mean).toFixed(5) : '-',
      units: data?.soil_moisture?.units,
    },
    {
      title: '🌱 Soil Moisture Max Hourly',
      value: data?.soil_moisture?.max_hourly_mean != null && !isNaN(data?.soil_moisture?.max_hourly_mean) ? Number(data?.soil_moisture?.max_hourly_mean).toFixed(5) : '-',
      units: data?.soil_moisture?.units,
    },
    {
      title: '🌱 Soil Moisture Range Hourly',
      value: data?.soil_moisture?.range_hourly_mean != null && !isNaN(data?.soil_moisture?.range_hourly_mean) ? Number(data?.soil_moisture?.range_hourly_mean).toFixed(5) : '-',
      units: data?.soil_moisture?.units,
    },
    {
      title: '🌱 Soil Temperature Monthly Mean',
      value: data?.soil_temperature?.monthly_mean != null && !isNaN(data?.soil_temperature?.monthly_mean) ? Number(data?.soil_temperature?.monthly_mean).toFixed(5) : '-',
      units: data?.soil_temperature?.units,
    },
    {
      title: '🌱 Soil Temperature Min Hourly',
      value: data?.soil_temperature?.min_hourly_mean != null && !isNaN(data?.soil_temperature?.min_hourly_mean) ? Number(data?.soil_temperature?.min_hourly_mean).toFixed(5) : '-',
      units: data?.soil_temperature?.units,
    },
    {
      title: '🌱 Soil Temperature Max Hourly',
      value: data?.soil_temperature?.max_hourly_mean != null && !isNaN(data?.soil_temperature?.max_hourly_mean) ? Number(data?.soil_temperature?.max_hourly_mean).toFixed(5) : '-',
      units: data?.soil_temperature?.units,
    },
    {
      title: '🌱 Soil Temperature Range Hourly',
      value: data?.soil_temperature?.range_hourly_mean != null && !isNaN(data?.soil_temperature?.range_hourly_mean) ? Number(data?.soil_temperature?.range_hourly_mean).toFixed(5) : '-',
      units: data?.soil_temperature?.units,
    },
    {
      title: '🌿 NDWI Mean',
      value: data?.ndwi?.monthly_mean != null && !isNaN(data?.ndwi?.monthly_mean) ? Number(data?.ndwi?.monthly_mean).toFixed(5) : '-',
      units: null,
    },
    {
      title: '🌿 NDVI Mean',
      value: data?.ndvi?.monthly_mean != null && !isNaN(data?.ndvi?.monthly_mean) ? Number(data?.ndvi?.monthly_mean).toFixed(5) : '-',
      units: null,
    },
    ...Object.entries(landCoverClasses).map(([name, val]) => ({
      title: `🏞️ Land Cover ${name} Percentage`,
      value: val != null && !isNaN(val) ? `${Number(val).toFixed(2)}%` : '-',
      units: null,
    })),
    {
      title: '🌲 Forest Loss Percentage',
      value: data?.forest_loss_percentage?.percent != null && !isNaN(data?.forest_loss_percentage?.percent) ? `${Number(data?.forest_loss_percentage?.percent).toFixed(2)}%` : '-',
      units: null,
    },
    {
      title: '👥 Population Density',
      value: data?.population_density != null && !isNaN(data?.population_density) ? Number(data?.population_density).toFixed(2) : '-',
      units: 'persons/km²',
    },
    {
      title: '💰 GDP',
      value: data?.gdp != null && !isNaN(data?.gdp) ? Number(data?.gdp).toFixed(2) : '-',
      units: 'MIO_EUR',
    },
    {
      title: '👷 Employment',
      value: data?.employment != null && !isNaN(data?.employment) ? Number(data?.employment).toFixed(2) : '-',
      units: 'THS',
    },
    {
      title: '🐄 Livestock Population',
      value: data?.livestock != null && !isNaN(data?.livestock) ? Number(data?.livestock).toFixed(2) : '-',
      units: 'THS_HD',
    },
  ];
  */

  return (
    <main style={{ maxWidth: '100%', margin: '0 auto', padding: '24px 24px 24px 24px', fontFamily: 'Arial, sans-serif', backgroundColor: 'white', color: '#333' }}>
      <div style={{ width: '80%', margin: '0 auto', border: '1px solid #e0e0e0', borderRadius: '8px', backgroundColor: '#fff', padding: '16px' }}>
        <style>
          {`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}
        </style>
        <header style={{ textAlign: 'center', marginBottom: '32px', width: '100%', margin: '0 auto 32px auto' }}>
          <h1 style={{ margin: '0', color: '#333' }}>Leptospirosis in Europe: A Fine-Scale Assessment of Environmental and Socioeconomic Risk Drivers</h1>
        </header>

        <div style={{ display: 'flex', gap: '12px', marginBottom: '24px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <select
            value={selectedVersion}
            onChange={handleVersionChange}
            disabled={loading}
            style={{ padding: '8px 12px', border: '1px solid #ccc', borderRadius: '4px', backgroundColor: 'white', color: 'black' }}
          >
            {NUTS_VERSIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>

          <div style={{ width: '300px' }}>
            <Select
              options={regions}
              value={selectedRegion}
              onChange={setSelectedRegion}
              placeholder="Search NUTS3 region"
              isClearable
              isDisabled={loading || regions.length === 0}
              menuPortalTarget={document.body}
              styles={{
                control: (base) => ({ ...base, backgroundColor: 'white', color: 'black' }),
                menuPortal: (base) => ({ ...base, zIndex: 9999 }),
                menu: (base) => ({ ...base, zIndex: 9999 }),
                singleValue: (base) => ({ ...base, color: 'black' }),
                placeholder: (base) => ({ ...base, color: '#666' }),
                input: (base) => ({ ...base, color: 'black' }),
                option: (base) => ({ ...base, color: 'black' }),
              }}
              filterOption={(option, input) =>
                option.label.toLowerCase().includes(input.toLowerCase())
              }
            />
          </div>

          <select
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            disabled={loading}
            style={{ padding: '8px 12px', border: '1px solid #ccc', borderRadius: '4px', backgroundColor: 'white', color: 'black' }}
          >
            {yearOptions.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>

          <select
            value={month.value}
            onChange={(e) => setMonth(MONTHS.find((m) => m.value === e.target.value))}
            disabled={loading}
            style={{ padding: '8px 12px', border: '1px solid #ccc', borderRadius: '4px', backgroundColor: 'white', color: 'black' }}
          >
            {MONTHS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>

          <button
            onClick={handleFetchData}
            disabled={!selectedRegion || loading}
            style={{
              padding: '10px 16px',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            {loading ? 'Predicting…' : 'Predict'}
          </button>
        </div>

        <div style={{ padding: '16px', border: '1px solid #e0e0e0', borderRadius: '8px', backgroundColor: '#fff', textAlign: 'center', width: '100%', margin: '0 auto', marginBottom: '20px', boxSizing: 'border-box' }}>
          <h3 style={{ margin: '0 0 8px', color: '#333' }}>📊 Predicted Probability</h3>
          {loading ? (
            <div style={{ width: '20px', height: '20px', border: '2px solid #ccc', borderTop: '2px solid #007bff', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto' }}></div>
          ) : data ? (
            <p style={{ margin: '0', fontSize: '2rem', fontWeight: 'bold', color: 'black' }}>
              {data?.predicted_probability != null && !isNaN(data?.predicted_probability) ? (Number(data?.predicted_probability) * 100).toFixed(2) + '%' : '-'}
            </p>
          ) : (
            <p style={{ margin: '0', fontSize: '1.5rem', color: '#666' }}>Select a region and click Predict to see the probability</p>
          )}
        </div>

        {error && <div style={{ marginBottom: '16px', padding: '12px', border: '1px solid #dc3545', backgroundColor: '#f8d7da', color: '#721c24', borderRadius: '4px', width: '100%', margin: '0 auto' }}>{error}</div>}

        <div ref={mapRef} style={{ width: '100%', height: '500px', border: '1px solid #ccc', borderRadius: '8px', margin: '0 auto', marginBottom: '20px', position: 'relative' }}>
          <div style={{ position: 'absolute', top: '20px', right: '10px', fontSize: '14px', fontWeight: 'bold', color: '#333', zIndex: 1001 }}>Risk Level</div>
          <div style={{ position: 'absolute', top: '50px', right: '10px', width: '20px', height: '200px', background: 'linear-gradient(to top, blue 0%, white 63%, orange 100%)', border: '1px solid #ccc', zIndex: 1000 }}>
            {data && (
              <div style={{ position: 'absolute', bottom: '0', width: '100%', height: `${(data?.predicted_probability || 0) * 100}%`, background: 'transparent' }}></div>
            )}
            <div style={{ position: 'absolute', bottom: '0', right: '25px', fontSize: '10px', color: '#333', zIndex: 1001 }}>0%</div>
            <div style={{ position: 'absolute', top: '37%', right: '25px', transform: 'translateY(-50%)', fontSize: '10px', color: '#333', zIndex: 1001 }}>63%</div>
            <div style={{ position: 'absolute', top: '0', right: '25px', fontSize: '10px', color: '#333', zIndex: 1001 }}>100%</div>
          </div>
        </div>

        {/* Commented out legend and features sections */}
        {/*
        {data && (
          <div style={{ marginBottom: '24px', padding: '16px', border: '1px solid #e0e0e0', borderRadius: '8px', backgroundColor: '#fff', textAlign: 'center' }}>
            <strong>Legend:</strong><br />
            <span style={{ color: 'blue' }}>●</span> Low Risk (0-42%)<br />
            <span style={{ color: 'gray' }}>●</span> Medium Risk (42-84%)<br />
            <span style={{ color: 'orange' }}>●</span> High Risk (84-100%)
          </div>
        )}

        <div style={{ marginBottom: '24px', padding: '16px', border: '1px solid #e0e0e0', borderRadius: '8px', backgroundColor: '#fff', display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
          {features.map((feature, index) => (
            <div key={index} style={{ flex: '1 1 250px', padding: '12px', border: '1px solid #ddd', borderRadius: '4px', backgroundColor: '#f9f9f9' }}>
              <h4 style={{ margin: '0 0 8px', fontSize: '1rem', color: '#333' }}>
                {feature.title}
                {feature.units && <span style={{ fontWeight: 'normal', color: '#666', fontSize: '0.9rem' }}> ({feature.units})</span>}
              </h4>
              {loading ? (
                <div style={{ width: '20px', height: '20px', border: '2px solid #ccc', borderTop: '2px solid #007bff', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto' }}></div>
              ) : (
                <p style={{ margin: '0', fontSize: '1.1rem', fontWeight: 'bold' }}>{feature.value}</p>
              )}
            </div>
          ))}
        </div>
        */}
      </div>
    </main>
  );
}

export default App;