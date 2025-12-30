document.addEventListener('DOMContentLoaded', () => {
    const state = {
        granularity: 'daily',
        rangePreset: 'last30',
        history: [],
        chart: null
    };

    const els = {
        granularityToggle: document.getElementById('granularityToggle'),
        rangeSelect: document.getElementById('rangeSelect'),
        customRangeBlock: document.getElementById('customRangeBlock'),
        startDate: document.getElementById('startDate'),
        endDate: document.getElementById('endDate'),
        rangeTotal: document.getElementById('rangeTotal'),
        rangeWindow: document.getElementById('rangeWindow'),
        topDomain: document.getElementById('topDomain'),
        topDomainTime: document.getElementById('topDomainTime'),
        avgPerBucket: document.getElementById('avgPerBucket'),
        bucketMeta: document.getElementById('bucketMeta'),
        periodList: document.getElementById('periodList'),
        periodEmpty: document.getElementById('periodEmpty'),
        heroTotal: document.getElementById('heroTotal'),
        heroRange: document.getElementById('heroRange'),
        heroTopDomain: document.getElementById('heroTopDomain'),
        chartCanvas: document.getElementById('historyChart').getContext('2d'),
        chartEmpty: document.getElementById('chartEmpty')
    };

    initialize();

    function initialize() {
        const today = new Date();
        els.startDate.value = toInputDate(addDays(today, -29));
        els.endDate.value = toInputDate(today);

        els.granularityToggle.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('click', () => {
                if (btn.classList.contains('active')) {
                    return;
                }
                els.granularityToggle.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.granularity = btn.dataset.value;
                render();
            });
        });

        els.rangeSelect.addEventListener('change', () => {
            state.rangePreset = els.rangeSelect.value;
            const isCustom = state.rangePreset === 'custom';
            els.customRangeBlock.classList.toggle('visible', isCustom);
            render();
        });

        [els.startDate, els.endDate].forEach(input => {
            input.addEventListener('change', () => {
                if (state.rangePreset !== 'custom') {
                    return;
                }
                const start = toDate(els.startDate.value);
                const end = toDate(els.endDate.value);
                if (start && end && start > end) {
                    alert('The start date must be earlier than the end date.');
                    return;
                }
                render();
            });
        });

        chrome.storage.local.get(['timeHistory'], result => {
            state.history = normalizeHistory(result.timeHistory || {});
            updateHeroCard();
            render();
        });

        chrome.storage.onChanged.addListener((changes, area) => {
            if (area !== 'local' || !changes.timeHistory) {
                return;
            }
            state.history = normalizeHistory(changes.timeHistory.newValue || {});
            updateHeroCard();
            render();
        });
    }

    function render() {
        if (!state.history.length) {
            showEmptyState();
            return;
        }

        const { start, end } = resolveRange();
        const filtered = state.history.filter(entry => entry.date >= start && entry.date <= end);
        const grouped = groupEntries(filtered, state.granularity);
        updateSummary(grouped, start, end);
        drawChart(grouped);
        populatePeriods(grouped);
    }

    function showEmptyState() {
        els.rangeTotal.textContent = '--';
        els.rangeWindow.textContent = 'No tracking data yet';
        els.topDomain.textContent = '--';
        els.topDomainTime.textContent = '';
        els.avgPerBucket.textContent = '--';
        els.bucketMeta.textContent = 'Keep the tracker running to unlock insights';
        els.chartEmpty.hidden = false;
        els.periodEmpty.hidden = false;
        els.periodList.innerHTML = '';
    }

    function updateHeroCard() {
        if (!state.history.length) {
            els.heroTotal.textContent = '--h --m';
            els.heroRange.textContent = 'Tracking queue is empty';
            els.heroTopDomain.textContent = '';
            return;
        }
        const latest = state.history[state.history.length - 1];
        els.heroTotal.textContent = formatDuration(latest.totalSeconds, { compact: true });
        els.heroRange.textContent = formatDateRange(latest.date, latest.date);
        const top = getTopDomain(latest.domains);
        els.heroTopDomain.textContent = top ? `Top domain · ${top.name} (${formatDuration(top.value)})` : 'Focus not yet categorized';
    }

    function resolveRange() {
        const today = new Date();
        let start;
        let end;

        switch (state.rangePreset) {
            case 'last7':
                end = today;
                start = addDays(today, -6);
                break;
            case 'last30':
                end = today;
                start = addDays(today, -29);
                break;
            case 'last90':
                end = today;
                start = addDays(today, -89);
                break;
            case 'thisMonth':
                end = today;
                start = new Date(today.getFullYear(), today.getMonth(), 1);
                break;
            case 'custom':
                start = toDate(els.startDate.value) || addDays(today, -29);
                end = toDate(els.endDate.value) || today;
                break;
            default:
                end = today;
                start = addDays(today, -29);
        }

        if (start > end) {
            const temp = start;
            start = end;
            end = temp;
        }

        // Normalize to midnight boundaries
        start = new Date(start.getFullYear(), start.getMonth(), start.getDate());
        end = new Date(end.getFullYear(), end.getMonth(), end.getDate());
        return { start, end };
    }

    function updateSummary(grouped, start, end) {
        if (!grouped.length) {
            els.rangeTotal.textContent = '0h 00m';
            els.rangeWindow.textContent = formatDateRange(start, end);
            els.topDomain.textContent = '--';
            els.topDomainTime.textContent = 'No visits recorded';
            els.avgPerBucket.textContent = '0h 00m';
            els.bucketMeta.textContent = `0 ${state.granularity} buckets`;
            return;
        }

        const totalSeconds = grouped.reduce((sum, bucket) => sum + bucket.totalSeconds, 0);
        els.rangeTotal.textContent = formatDuration(totalSeconds, { includeSeconds: false });
        els.rangeWindow.textContent = formatDateRange(start, end);

        const aggregatedDomains = {};
        grouped.forEach(bucket => {
            Object.entries(bucket.domains).forEach(([domain, value]) => {
                aggregatedDomains[domain] = (aggregatedDomains[domain] || 0) + value;
            });
        });
        const top = getTopDomain(aggregatedDomains);
        els.topDomain.textContent = top ? top.name : '--';
        els.topDomainTime.textContent = top ? formatDuration(top.value) : 'No activity in range';

        const avg = totalSeconds / grouped.length;
        els.avgPerBucket.textContent = formatDuration(avg, { includeSeconds: false });
        els.bucketMeta.textContent = `${grouped.length} ${state.granularity} bucket${grouped.length === 1 ? '' : 's'}`;
    }

    function drawChart(grouped) {
        if (state.chart) {
            state.chart.destroy();
            state.chart = null;
        }

        if (!grouped.length) {
            els.chartEmpty.hidden = false;
            return;
        }
        els.chartEmpty.hidden = true;

        const labels = grouped.map(bucket => bucket.label);
        const values = grouped.map(bucket => +(bucket.totalSeconds / 3600).toFixed(2));

        state.chart = new Chart(els.chartCanvas, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Focused hours',
                    data: values,
                    fill: true,
                    tension: 0.35,
                    borderColor: 'rgba(244, 201, 111, 0.9)',
                    backgroundColor: 'rgba(244, 201, 111, 0.25)',
                    pointBackgroundColor: '#f4c96f',
                    pointBorderColor: '#0f172a'
                }]
            },
            options: {
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#e2e8f0' }
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const value = typeof ctx.parsed === 'object' ? ctx.parsed.y : ctx.parsed;
                                return `${value.toFixed(2)} h`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#cbd5f5' },
                        grid: { color: 'rgba(255,255,255,0.08)' }
                    },
                    y: {
                        ticks: { color: '#cbd5f5' },
                        grid: { color: 'rgba(255,255,255,0.08)' }
                    }
                }
            }
        });
    }

    function populatePeriods(grouped) {
        if (!grouped.length) {
            els.periodList.innerHTML = '';
            els.periodEmpty.hidden = false;
            return;
        }

        els.periodEmpty.hidden = true;
        els.periodList.innerHTML = '';

        grouped.forEach(bucket => {
            const card = document.createElement('div');
            card.className = 'period-card';

            const heading = document.createElement('h5');
            heading.textContent = bucket.label;
            card.appendChild(heading);

            const total = document.createElement('p');
            total.textContent = `Focused ${formatDuration(bucket.totalSeconds)} · ${formatDateRange(bucket.start, bucket.end)}`;
            card.appendChild(total);

            const top = getTopDomain(bucket.domains);
            if (top) {
                const pill = document.createElement('div');
                pill.className = 'pill';
                pill.textContent = `${top.name} · ${formatDuration(top.value, { includeSeconds: false })}`;
                card.appendChild(pill);
            }

            els.periodList.appendChild(card);
        });
    }

    function groupEntries(entries, granularity) {
        const buckets = {};
        entries.forEach(entry => {
            const descriptor = getBucketDescriptor(entry.date, granularity);
            if (!descriptor) {
                return;
            }
            if (!buckets[descriptor.key]) {
                buckets[descriptor.key] = {
                    ...descriptor,
                    totalSeconds: 0,
                    domains: {}
                };
            }
            buckets[descriptor.key].totalSeconds += entry.totalSeconds;
            Object.entries(entry.domains || {}).forEach(([domain, value]) => {
                buckets[descriptor.key].domains[domain] = (buckets[descriptor.key].domains[domain] || 0) + value;
            });
        });

        return Object.values(buckets).sort((a, b) => a.start - b.start);
    }

    function getBucketDescriptor(date, granularity) {
        const d = new Date(date.getFullYear(), date.getMonth(), date.getDate());
        if (granularity === 'daily') {
            return {
                key: d.toISOString(),
                label: d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
                start: d,
                end: d
            };
        }

        if (granularity === 'weekly') {
            const start = startOfWeek(d);
            const end = addDays(start, 6);
            return {
                key: `week-${start.toISOString()}`,
                label: `Week ${isoWeekNumber(d)} · ${formatShort(start)} – ${formatShort(end)}`,
                start,
                end
            };
        }

        if (granularity === 'monthly') {
            const start = new Date(d.getFullYear(), d.getMonth(), 1);
            const end = new Date(d.getFullYear(), d.getMonth() + 1, 0);
            return {
                key: `month-${start.getFullYear()}-${start.getMonth()}`,
                label: start.toLocaleDateString(undefined, { month: 'long', year: 'numeric' }),
                start,
                end
            };
        }
        return null;
    }

    function normalizeHistory(raw) {
        return Object.entries(raw).map(([dateKey, entry]) => {
            const safeEntry = entry && typeof entry === 'object' ? entry : {};
            const date = new Date(`${dateKey}T00:00:00`);
            const domains = safeEntry.domains && typeof safeEntry.domains === 'object' ? safeEntry.domains : {};
            const explicit = typeof safeEntry.totalSeconds === 'number' ? safeEntry.totalSeconds : null;
            const fallback = Object.values(domains).reduce((sum, value) => sum + (value || 0), 0);
            return {
                dateKey,
                date,
                totalSeconds: explicit !== null ? explicit : fallback,
                domains
            };
        }).filter(item => !Number.isNaN(item.date.getTime()))
          .sort((a, b) => a.date - b.date);
    }

    function formatDuration(seconds, options = {}) {
        if (!Number.isFinite(seconds) || seconds <= 0) {
            return options.compact ? '0h 00m' : '0h 00m 00s';
        }
        const includeSeconds = options.includeSeconds !== false;
        const totalSeconds = Math.round(seconds);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const secs = totalSeconds % 60;
        if (options.compact) {
            return `${hours}h ${minutes}m`;
        }
        return includeSeconds ? `${hours}h ${minutes.toString().padStart(2, '0')}m ${secs.toString().padStart(2, '0')}s` : `${hours}h ${minutes.toString().padStart(2, '0')}m`;
    }

    function formatDateRange(start, end) {
        if (!start || !end) {
            return '';
        }
        const formatter = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
        if (start.toDateString() === end.toDateString()) {
            return formatter.format(start);
        }
        return `${formatter.format(start)} – ${formatter.format(end)}`;
    }

    function formatShort(date) {
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }

    function startOfWeek(date) {
        const clone = new Date(date);
        const day = clone.getDay();
        const diff = (day === 0 ? -6 : 1 - day);
        clone.setDate(clone.getDate() + diff);
        return new Date(clone.getFullYear(), clone.getMonth(), clone.getDate());
    }

    function isoWeekNumber(date) {
        const target = new Date(date.valueOf());
        const dayNr = (date.getDay() + 6) % 7;
        target.setDate(target.getDate() - dayNr + 3);
        const firstThursday = new Date(target.getFullYear(), 0, 4);
        const weekNumber = 1 + Math.round(((target - firstThursday) / 86400000 - 3) / 7);
        return weekNumber;
    }

    function getTopDomain(domains = {}) {
        const entries = Object.entries(domains);
        if (!entries.length) {
            return null;
        }
        const [name, value] = entries.sort((a, b) => b[1] - a[1])[0];
        return { name, value };
    }

    function addDays(date, amount) {
        const copy = new Date(date);
        copy.setDate(copy.getDate() + amount);
        return copy;
    }

    function toInputDate(date) {
        if (!date) {
            return '';
        }
        const y = date.getFullYear();
        const m = String(date.getMonth() + 1).padStart(2, '0');
        const d = String(date.getDate()).padStart(2, '0');
        return `${y}-${m}-${d}`;
    }

    function toDate(value) {
        if (!value) {
            return null;
        }
        const date = new Date(value);
        return Number.isNaN(date.getTime()) ? null : date;
    }
});
