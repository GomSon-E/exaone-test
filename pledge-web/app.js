// app.js
document.addEventListener('DOMContentLoaded', function() {
    // API 엔드포인트 설정 (로컬 경로로 변경)
    const API_BASE_URL = ''; // 같은 서버에서 제공되므로 상대 경로 사용
    
    // DOM 요소
    const refreshBtn = document.getElementById('refreshBtn');
    const updateStatus = document.getElementById('updateStatus');
    const totalResponses = document.getElementById('totalResponses');
    const todayResponses = document.getElementById('todayResponses');
    const avgResponseTime = document.getElementById('avgResponseTime');
    const tableHead = document.getElementById('tableHead');
    const tableBody = document.getElementById('tableBody');
    
    // 차트 객체 참조
    let responseChart = null;
    
    // 초기 데이터 로드
    loadData();
    
    // 새로고침 버튼 이벤트
    refreshBtn.addEventListener('click', function() {
        loadData(true);
    });
    
    // 5분마다 자동 새로고침
    setInterval(() => loadData(), 300000);
    
    // 데이터 로드 함수
    function loadData(forceRefresh = false) {
        // 로딩 상태 표시
        updateStatus.innerHTML = `<span class="refreshing-indicator"></span> 데이터를 가져오는 중...`;
        refreshBtn.disabled = true;
        
        // API 호출
        fetch(`${API_BASE_URL}/api/responses?force_refresh=${forceRefresh}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP 오류! 상태: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // 응답 데이터 처리
                displayData(data);
                
                // 통계 데이터 가져오기
                return fetch(`${API_BASE_URL}/api/stats`);
            })
            .then(response => response.json())
            .then(stats => {
                displayStats(stats);
                
                // 업데이트 완료 메시지
                if (data.last_updated) {
                    const lastUpdated = new Date(data.last_updated);
                    updateStatus.textContent = `마지막 업데이트: ${lastUpdated.toLocaleString()}`;
                } else {
                    updateStatus.textContent = '데이터가 업데이트되었습니다.';
                }
                
                refreshBtn.disabled = false;
            })
            .catch(error => {
                console.error('Error fetching data:', error);
                updateStatus.textContent = `오류 발생: ${error.message}`;
                refreshBtn.disabled = false;
            });
    }
    
    // 응답 데이터 표시 함수
    function displayData(responseData) {
        const data = responseData.data;
        
        if (!data || data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="100%" class="text-center py-3">데이터가 없습니다</td></tr>';
            return;
        }
        
        // 테이블 헤더 생성
        const headers = Object.keys(data[0]);
        tableHead.innerHTML = '';
        const headerRow = document.createElement('tr');
        
        headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        
        tableHead.appendChild(headerRow);
        
        // 테이블 데이터 행 생성 (최신 10개)
        tableBody.innerHTML = '';
        const recentData = data.slice(-10).reverse();
        
        recentData.forEach(row => {
            const tr = document.createElement('tr');
            
            headers.forEach(header => {
                const td = document.createElement('td');
                td.textContent = row[header] || '';
                tr.appendChild(td);
            });
            
            tableBody.appendChild(tr);
        });
        
        // 차트 데이터 생성 및 업데이트
        updateChart(data);
    }
    
    // 통계 데이터 표시 함수
    function displayStats(stats) {
        if (stats.error) {
            console.error('통계 데이터 오류:', stats.error);
            return;
        }
        
        totalResponses.textContent = stats.total_responses || 0;
        todayResponses.textContent = stats.today_responses || 0;
        
        // 여기에 추가 통계 표시 (예: 응답 시간 계산)
        calculateAverageResponseTime(stats);
    }
    
    // 평균 응답 시간 계산 (예시)
    function calculateAverageResponseTime(stats) {
        // 실제 구현은 응답 데이터 구조에 따라 다를 수 있습니다
        avgResponseTime.textContent = '2.5분'; // 예시 값
    }
    
    // 차트 업데이트 함수
    function updateChart(data) {
        // 타임스탬프 필드 찾기
        const timestampField = Object.keys(data[0]).find(key => 
            key.toLowerCase().includes('timestamp') || key.toLowerCase().includes('타임스탬프'));
        
        if (!timestampField) return;
        
        // 날짜별 응답 개수 집계
        const dateMap = new Map();
        
        data.forEach(entry => {
            try {
                // 날짜 변환 (구글 폼 형식에 따라 변경될 수 있음)
                const date = new Date(entry[timestampField]);
                const dateString = date.toLocaleDateString();
                
                dateMap.set(dateString, (dateMap.get(dateString) || 0) + 1);
            } catch (e) {
                console.warn('날짜 파싱 오류:', e);
            }
        });
        
        // 차트 데이터 준비
        const chartDates = Array.from(dateMap.keys()).slice(-7); // 최근 7일
        const chartValues = chartDates.map(date => dateMap.get(date) || 0);
        
        // 차트 생성 또는 업데이트
        const ctx = document.getElementById('responseChart').getContext('2d');
        
        if (responseChart) {
            responseChart.data.labels = chartDates;
            responseChart.data.datasets[0].data = chartValues;
            responseChart.update();
        } else {
            responseChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartDates,
                    datasets: [{
                        label: '일별 응답 수',
                        data: chartValues,
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                precision: 0
                            }
                        }
                    }
                }
            });
        }
    }
});
