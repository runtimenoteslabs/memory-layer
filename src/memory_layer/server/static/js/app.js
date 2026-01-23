/**
 * Memory Layer Web UI Application
 */

class MemoryLayerApp {
    constructor() {
        this.currentView = 'dashboard';
        this.memories = [];
        this.currentPage = 1;
        this.pageSize = 50;
        this.selectedMemory = null;
        this.stats = null;

        this.init();
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    async init() {
        this.setupEventListeners();
        this.loadTheme();
        await this.loadDashboard();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.showView(e.target.dataset.view);
            });
        });

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => {
            this.toggleTheme();
        });

        // Add memory form
        document.getElementById('add-memory-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addMemory();
        });

        // Search on enter
        document.getElementById('search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });

        // Filters
        document.getElementById('filter-category').addEventListener('change', () => {
            this.loadMemories();
        });
        document.getElementById('filter-project').addEventListener('change', () => {
            this.loadMemories();
        });
        document.getElementById('filter-search').addEventListener('input', () => {
            this.filterMemoriesLocally();
        });

        // Close modal on background click
        document.getElementById('memory-modal').addEventListener('click', (e) => {
            if (e.target.id === 'memory-modal') {
                this.closeModal();
            }
        });

        // Escape key closes modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }

    // =========================================================================
    // Theme Management
    // =========================================================================

    loadTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    }

    toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        this.updateThemeIcon(next);
    }

    updateThemeIcon(theme) {
        const icon = document.querySelector('.theme-icon');
        icon.textContent = theme === 'dark' ? '\u263E' : '\u2600';
    }

    // =========================================================================
    // View Navigation
    // =========================================================================

    showView(viewName) {
        this.currentView = viewName;

        // Update nav buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === viewName);
        });

        // Show/hide views
        document.querySelectorAll('.view').forEach(view => {
            view.classList.toggle('active', view.id === `view-${viewName}`);
        });

        // Load data for view
        switch (viewName) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'memories':
                this.loadMemories();
                break;
            case 'search':
                document.getElementById('search-input').focus();
                break;
            case 'beads':
                this.loadBeads();
                break;
        }
    }

    // =========================================================================
    // Dashboard
    // =========================================================================

    async loadDashboard() {
        try {
            this.stats = await api.getStats();
            this.renderStats();
            await this.loadRecentMemories();
        } catch (error) {
            this.showToast('Failed to load dashboard', 'error');
        }
    }

    renderStats() {
        const stats = this.stats;
        document.getElementById('stat-total').textContent = stats.total_memories || 0;
        document.getElementById('stat-active').textContent = stats.active_memories || 0;
        document.getElementById('stat-archived').textContent = stats.archived_memories || 0;
        document.getElementById('stat-avg-score').textContent =
            (stats.avg_outcome_score || 0).toFixed(2);

        // Render category bars
        this.renderCategoryBars(stats.by_category || {});
    }

    renderCategoryBars(byCategory) {
        const container = document.getElementById('category-bars');
        const maxCount = Math.max(...Object.values(byCategory), 1);

        // All categories with their colors (matching CSS)
        const categoryColors = {
            'architecture': '#3b82f6',    // blue
            'convention': '#8b5cf6',      // purple
            'decision': '#06b6d4',        // cyan
            'pattern': '#10b981',         // green (success)
            'gotcha': '#ef4444',          // red (danger)
            'workaround': '#f97316',      // orange
            'troubleshooting': '#f59e0b', // amber (warning)
            'command': '#64748b',         // slate
            'preference': '#ec4899',      // pink
            'general': '#6b7280'          // gray
        };

        const categories = Object.keys(categoryColors);

        container.innerHTML = categories
            .map(cat => {
                const count = byCategory[cat] || 0;
                const width = maxCount > 0 ? (count / maxCount * 100).toFixed(1) : 0;
                const color = categoryColors[cat];
                return `
                    <div class="category-bar">
                        <span class="label">${cat}</span>
                        <div class="bar-container">
                            <div class="bar" style="width: ${width}%; background-color: ${color}"></div>
                        </div>
                        <span class="count">${count}</span>
                    </div>
                `;
            })
            .join('');
    }

    async loadRecentMemories() {
        try {
            const memories = await api.getMemories({ limit: 5 });
            this.renderRecentMemories(memories);
        } catch (error) {
            console.error('Failed to load recent memories', error);
        }
    }

    renderRecentMemories(memories) {
        const container = document.getElementById('recent-memories');

        if (!memories || memories.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No memories yet</p></div>';
            return;
        }

        container.innerHTML = memories.map(m => `
            <div class="recent-item" onclick="app.viewMemory('${m.id}')">
                <div class="content">${this.escapeHtml(m.content)}</div>
                <div class="meta">
                    <span class="category-badge ${m.category}">${m.category}</span>
                    <span>${this.formatDate(m.created_at)}</span>
                </div>
            </div>
        `).join('');
    }

    // =========================================================================
    // Memories List
    // =========================================================================

    async loadMemories() {
        try {
            const category = document.getElementById('filter-category').value;
            const project = document.getElementById('filter-project').value;

            console.log('Loading memories...');
            this.memories = await api.getMemories({
                category: category || undefined,
                project: project || undefined,
                limit: 100
            });
            console.log('Loaded memories:', this.memories);

            this.renderMemoriesTable();
            this.updateProjectFilter();
        } catch (error) {
            console.error('Failed to load memories:', error);
            const msg = error.message || JSON.stringify(error);
            this.showToast('Failed to load memories: ' + msg, 'error');
        }
    }

    renderMemoriesTable() {
        const tbody = document.getElementById('memories-tbody');
        const start = (this.currentPage - 1) * this.pageSize;
        const pageMemories = this.memories.slice(start, start + this.pageSize);

        if (pageMemories.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="empty-state">
                        <p>No memories found</p>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = pageMemories.map(m => `
            <tr onclick="app.viewMemory('${m.id}')">
                <td><span class="category-badge ${m.category}">${m.category}</span></td>
                <td class="content-cell">${this.escapeHtml(m.content)}</td>
                <td class="score-cell ${m.outcome_score > 0 ? 'positive' : m.outcome_score < 0 ? 'negative' : ''}">
                    ${m.outcome_score.toFixed(2)}
                </td>
                <td>${m.project || 'global'}</td>
                <td>${this.formatDate(m.created_at)}</td>
                <td class="actions-cell">
                    <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation(); app.viewMemory('${m.id}')">
                        View
                    </button>
                </td>
            </tr>
        `).join('');

        this.renderPagination();
    }

    renderPagination() {
        const totalPages = Math.ceil(this.memories.length / this.pageSize);
        const container = document.getElementById('pagination');

        if (totalPages <= 1) {
            container.innerHTML = '';
            return;
        }

        let html = '';
        for (let i = 1; i <= totalPages; i++) {
            html += `<button class="${i === this.currentPage ? 'active' : ''}"
                       onclick="app.goToPage(${i})">${i}</button>`;
        }
        container.innerHTML = html;
    }

    goToPage(page) {
        this.currentPage = page;
        this.renderMemoriesTable();
    }

    updateProjectFilter() {
        const projects = [...new Set(this.memories.map(m => m.project).filter(Boolean))];
        const select = document.getElementById('filter-project');
        const current = select.value;

        select.innerHTML = '<option value="">All Projects</option>' +
            projects.map(p => `<option value="${p}">${p}</option>`).join('');

        select.value = current;
    }

    filterMemoriesLocally() {
        const searchTerm = document.getElementById('filter-search').value.toLowerCase();
        const rows = document.querySelectorAll('#memories-tbody tr');

        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(searchTerm) ? '' : 'none';
        });
    }

    // =========================================================================
    // Search
    // =========================================================================

    async performSearch() {
        const query = document.getElementById('search-input').value.trim();
        if (!query) return;

        const categories = Array.from(document.querySelectorAll('.search-cat:checked'))
            .map(cb => cb.value);
        const searchType = document.getElementById('search-type').value;

        const container = document.getElementById('search-results');
        container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

        console.log('Performing search:', { query, categories, searchType });

        try {
            const results = await api.search(query, {
                categories: categories.length > 0 ? categories : null,
                limit: 20,
                searchType: searchType
            });

            console.log('Search results:', results);
            this.renderSearchResults(results, searchType);
        } catch (error) {
            console.error('Search failed:', error);
            container.innerHTML = '<div class="empty-state"><p>Search failed: ' + (error.message || 'Unknown error') + '</p></div>';
            this.showToast('Search failed: ' + (error.message || 'Unknown error'), 'error');
        }
    }

    renderSearchResults(results, searchType = 'semantic') {
        const container = document.getElementById('search-results');

        if (!results || results.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No results found</p></div>';
            return;
        }

        container.innerHTML = results.map(r => `
            <div class="search-result" onclick="app.viewMemory('${r.memory.id}')">
                <div class="header">
                    <span class="category-badge ${r.memory.category}">${r.memory.category}</span>
                    ${searchType === 'semantic'
                        ? `<span class="relevance-score">${(r.score * 100).toFixed(0)}% match</span>`
                        : `<span class="relevance-score">Keyword match</span>`
                    }
                </div>
                <div class="content">${this.escapeHtml(r.memory.content)}</div>
                <div class="meta">
                    <span>Score: ${r.memory.outcome_score.toFixed(2)}</span>
                    <span>${r.memory.project || 'global'}</span>
                    <span>${this.formatDate(r.memory.created_at)}</span>
                </div>
            </div>
        `).join('');
    }

    // =========================================================================
    // Add Memory
    // =========================================================================

    async addMemory() {
        const content = document.getElementById('memory-content').value.trim();
        const category = document.getElementById('memory-category').value;
        const project = document.getElementById('memory-project').value.trim() || null;
        const tagsInput = document.getElementById('memory-tags').value.trim();
        const tags = tagsInput ? tagsInput.split(',').map(t => t.trim()).filter(Boolean) : [];

        if (!content) {
            this.showToast('Content is required', 'warning');
            return;
        }

        try {
            await api.createMemory({
                content,
                category,
                project,
                tags
            });

            this.showToast('Memory saved successfully', 'success');
            document.getElementById('add-memory-form').reset();
            this.showView('memories');
        } catch (error) {
            this.showToast('Failed to save memory: ' + error.message, 'error');
        }
    }

    // =========================================================================
    // Memory Detail Modal
    // =========================================================================

    async viewMemory(id) {
        try {
            const memory = await api.getMemory(id);
            this.selectedMemory = memory;
            this.renderMemoryDetail(memory);
            document.getElementById('memory-modal').classList.add('active');
        } catch (error) {
            this.showToast('Failed to load memory', 'error');
        }
    }

    renderMemoryDetail(memory) {
        const body = document.getElementById('modal-body');
        body.innerHTML = `
            <div class="detail-row">
                <div class="detail-label">Category</div>
                <div class="detail-value">
                    <span class="category-badge ${memory.category}">${memory.category}</span>
                </div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Content</div>
                <div class="detail-content">${this.escapeHtml(memory.content)}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Outcome Score</div>
                <div class="detail-value">${memory.outcome_score.toFixed(2)}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Project</div>
                <div class="detail-value">${memory.project || 'global'}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Tags</div>
                <div class="detail-value">${memory.tags?.join(', ') || 'None'}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">Created</div>
                <div class="detail-value">${this.formatDate(memory.created_at)}</div>
            </div>
            <div class="detail-row">
                <div class="detail-label">ID</div>
                <div class="detail-value" style="font-family: monospace; font-size: 0.8rem;">${memory.id}</div>
            </div>
        `;
    }

    closeModal() {
        document.getElementById('memory-modal').classList.remove('active');
        this.selectedMemory = null;
    }

    async recordOutcome(outcome) {
        if (!this.selectedMemory) return;

        try {
            await api.recordOutcome(this.selectedMemory.id, outcome);
            this.showToast(`Recorded outcome: ${outcome}`, 'success');
            this.closeModal();
            await this.loadDashboard();
        } catch (error) {
            this.showToast('Failed to record outcome', 'error');
        }
    }

    async deleteMemory() {
        if (!this.selectedMemory) return;

        if (!confirm('Are you sure you want to delete this memory?')) {
            return;
        }

        try {
            await api.deleteMemory(this.selectedMemory.id);
            this.showToast('Memory deleted', 'success');
            this.closeModal();
            await this.loadMemories();
            await this.loadDashboard();
        } catch (error) {
            this.showToast('Failed to delete memory', 'error');
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    async exportMemories() {
        try {
            const memories = await api.exportMemories();
            const json = JSON.stringify(memories, null, 2);
            const blob = new Blob([json], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `memory-layer-export-${new Date().toISOString().split('T')[0]}.json`;
            a.click();

            URL.revokeObjectURL(url);
            this.showToast('Export complete', 'success');
        } catch (error) {
            this.showToast('Export failed', 'error');
        }
    }

    // =========================================================================
    // Unified Tasks Integration (Phase 7 - Beads + Claude Code)
    // =========================================================================

    async loadBeads() {
        // Now uses unified tasks API
        try {
            const [statsResponse, tasksResponse] = await Promise.all([
                api.getTasksStats(),
                api.getTasks({ limit: 100 })
            ]);

            this.renderTasksStats(statsResponse);
            this.renderTasks(tasksResponse.tasks || [], tasksResponse.sources || []);
        } catch (error) {
            console.error('Failed to load tasks:', error);
            this.showToast('Failed to load tasks: ' + (error.message || 'Unknown error'), 'error');
        }
    }

    renderTasksStats(stats) {
        // Combine stats from all sources
        let total = 0;
        let inProgress = 0;
        let pending = 0;
        let done = 0;

        // Add Beads stats
        if (stats.beads && stats.beads.tasks) {
            const beadsStats = stats.beads.tasks.by_status || {};
            total += stats.beads.tasks.total_tasks || 0;
            inProgress += beadsStats.in_progress || 0;
            pending += beadsStats.pending || 0;
            done += beadsStats.done || 0;
        }

        // Add Claude Code stats
        if (stats.claude_code && stats.claude_code.tasks) {
            const ccStats = stats.claude_code.tasks.by_status || {};
            total += stats.claude_code.tasks.total_tasks || 0;
            inProgress += ccStats.in_progress || 0;
            pending += ccStats.pending || 0;
            done += ccStats.completed || 0;
        }

        document.getElementById('beads-total').textContent = total;
        document.getElementById('beads-in-progress').textContent = inProgress;
        document.getElementById('beads-pending').textContent = pending;
        document.getElementById('beads-done').textContent = done;

        // Update sources info
        const sourcesInfo = document.getElementById('tasks-sources');
        if (sourcesInfo) {
            const sources = stats.available_sources || [];
            sourcesInfo.textContent = sources.length > 0 ? `Sources: ${sources.join(', ')}` : 'No task sources available';
        }
    }

    renderTasks(tasks, sources) {
        const container = document.getElementById('beads-task-list');

        if (!tasks || tasks.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No tasks found</p>
                    <p class="text-muted">${sources.length > 0 ? `Available sources: ${sources.join(', ')}` : 'No task sources available'}</p>
                </div>
            `;
            return;
        }

        const statusColors = {
            'done': 'success',
            'completed': 'success',
            'in_progress': 'warning',
            'pending': 'secondary',
            'blocked': 'danger',
            'cancelled': 'danger'
        };

        const statusIcons = {
            'done': '✓',
            'completed': '✓',
            'in_progress': '►',
            'pending': '○',
            'blocked': '⊘',
            'cancelled': '✗'
        };

        const sourceLabels = {
            'beads': 'B',
            'claude_code': 'C'
        };

        const sourceColors = {
            'beads': 'source-beads',
            'claude_code': 'source-claude'
        };

        container.innerHTML = tasks.map(task => `
            <div class="task-item ${task.status}" onclick="app.showTaskContext('${task.id}', '${task.source || ''}')">
                <div class="task-status">
                    <span class="status-icon ${statusColors[task.status] || 'secondary'}">${statusIcons[task.status] || '?'}</span>
                </div>
                <div class="task-source">
                    <span class="source-badge ${sourceColors[task.source] || ''}" title="${task.source || 'unknown'}">${sourceLabels[task.source] || '?'}</span>
                </div>
                <div class="task-info">
                    <div class="task-title">${this.escapeHtml(task.title)}</div>
                    <div class="task-meta">
                        <span class="task-id">${task.id}</span>
                        <span class="task-status-text">${(task.status || '').replace('_', ' ')}</span>
                    </div>
                </div>
                <div class="task-actions">
                    <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation(); app.showTaskContext('${task.id}', '${task.source || ''}')">
                        View Context
                    </button>
                </div>
            </div>
        `).join('');
    }

    async showTaskContext(taskId, source = null) {
        try {
            // Use unified tasks API
            const context = await api.getTaskContext(taskId, source);

            document.getElementById('task-context').style.display = 'block';
            document.getElementById('task-context-title').textContent = `Context: ${context.task_title}`;

            const content = document.getElementById('task-context-content');

            if (context.memories_count === 0) {
                content.innerHTML = '<div class="empty-state"><p>No relevant memories found for this task</p></div>';
            } else {
                // Parse the formatted markdown-like content
                content.innerHTML = `
                    <div class="context-header">
                        <span class="task-status-badge ${context.task_status}">${(context.task_status || '').replace('_', ' ')}</span>
                        <span class="task-source-badge">${context.source || 'unknown'}</span>
                        <span class="memories-count">${context.memories_count} relevant memories</span>
                    </div>
                    <div class="context-memories">
                        ${this.formatContextMemories(context.formatted)}
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load task context:', error);
            this.showToast('Failed to load task context', 'error');
        }
    }

    formatContextMemories(formatted) {
        // Extract memory lines from formatted text
        const lines = formatted.split('\n').filter(line => line.startsWith('- **['));

        if (lines.length === 0) {
            return '<p>No memories in context</p>';
        }

        return lines.map(line => {
            // Parse: - **[category]** content...
            const match = line.match(/- \*\*\[(\w+)\]\*\* (.+)/);
            if (match) {
                const [, category, content] = match;
                return `
                    <div class="context-memory">
                        <span class="category-badge ${category}">${category}</span>
                        <span class="memory-content">${this.escapeHtml(content)}</span>
                    </div>
                `;
            }
            return '';
        }).join('');
    }

    closeTaskContext() {
        document.getElementById('task-context').style.display = 'none';
    }

    async syncBeads() {
        // Now uses unified tasks API
        try {
            this.showToast('Syncing tasks...', 'info');
            const result = await api.syncTasks();

            if (result.success) {
                const synced = result.total_tasks_synced || result.tasks_synced || 0;
                const outcomes = result.total_outcomes_recorded || result.outcomes_recorded || 0;
                this.showToast(`Synced: ${synced} tasks, ${outcomes} outcomes`, 'success');
                await this.loadBeads();
            } else {
                this.showToast('Sync failed', 'error');
            }
        } catch (error) {
            console.error('Sync failed:', error);
            this.showToast('Sync failed: ' + (error.message || 'Unknown error'), 'error');
        }
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatDate(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
}

// Initialize app
const app = new MemoryLayerApp();
