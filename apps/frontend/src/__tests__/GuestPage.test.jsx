import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

const mockListGuests = vi.hoisted(() => vi.fn());
const mockCreateGuest = vi.hoisted(() => vi.fn());
const mockUpdateGuest = vi.hoisted(() => vi.fn());
const mockDeleteGuest = vi.hoisted(() => vi.fn());

vi.mock('../api', () => ({
  listGuests: mockListGuests,
  createGuest: mockCreateGuest,
  updateGuest: mockUpdateGuest,
  deleteGuest: mockDeleteGuest,
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useOutletContext: () => ({
      runs: { apiKey: 'test-key' },
      theme: { isDark: false },
      auth: { handleAuthFailure: vi.fn() },
    }),
  };
});

import GuestPage from '../components/GuestPage';

describe('GuestPage interactions', () => {
  beforeEach(() => {
    mockListGuests.mockReset();
    mockListGuests.mockResolvedValue({
      items: [
        {
          id: 1,
          first_name: 'Ayse',
          last_name: 'Demir',
          email: 'ayse@example.com',
          phone: null,
          nationality: 'TUR',
          hotel: 'City Hotel',
          lead_time: 20,
          deposit_type: 'No Deposit',
          market_segment: 'Online TA',
          adults: 2,
          children: 0,
          babies: 0,
          stays_in_week_nights: 2,
          stays_in_weekend_nights: 1,
          previous_cancellations: 0,
          is_repeated_guest: 0,
          adr: 100,
          vip_status: false,
          risk_label: 'low',
          risk_score: 0.21,
          notes: null,
          created_at: '2026-03-11T08:00:00Z',
          updated_at: '2026-03-11T08:00:00Z',
        },
      ],
      total: 1,
    });

    mockCreateGuest.mockReset();
    mockCreateGuest.mockResolvedValue({
      id: 2,
      first_name: 'Yeni',
      last_name: 'Misafir',
      email: null,
      phone: null,
      nationality: null,
      identity_no: null,
      birth_date: null,
      gender: null,
      vip_status: false,
      notes: null,
      hotel: 'Resort Hotel',
      lead_time: 30,
      deposit_type: 'No Deposit',
      market_segment: 'Online TA',
      adults: 2,
      children: 0,
      babies: 0,
      stays_in_week_nights: 3,
      stays_in_weekend_nights: 1,
      is_repeated_guest: 0,
      previous_cancellations: 0,
      adr: 100,
      risk_label: 'medium',
      risk_score: 0.5,
      created_at: '2026-03-11T08:00:00Z',
      updated_at: '2026-03-11T08:00:00Z',
    });

    mockUpdateGuest.mockReset();
    mockDeleteGuest.mockReset();
  });

  it('loads guests and triggers filtered fetch on search interaction', async () => {
    const user = userEvent.setup();

    render(
      <MemoryRouter>
        <GuestPage />
      </MemoryRouter>,
    );

    expect(await screen.findByText(/Ayse Demir/i)).toBeInTheDocument();

    const searchInput = screen.getByLabelText(/Misafir ara/i);
    await user.clear(searchInput);
    await user.type(searchInput, 'Ayse');

    await waitFor(() => {
      const hasSearchCall = mockListGuests.mock.calls.some(([params]) => params?.search === 'Ayse');
      expect(hasSearchCall).toBe(true);
    });
  });

  it('opens create panel and submits new guest form', async () => {
    const user = userEvent.setup();

    render(
      <MemoryRouter>
        <GuestPage />
      </MemoryRouter>,
    );

    await screen.findByText(/Ayse Demir/i);

    await user.click(screen.getByRole('button', { name: /Yeni Misafir/i }));

    const firstNameInput = screen.getAllByLabelText(/Ad \*/i)[0];
    const lastNameInput = screen.getAllByLabelText(/Soyad \*/i)[0];

    await user.clear(firstNameInput);
    await user.type(firstNameInput, 'Yeni');
    await user.clear(lastNameInput);
    await user.type(lastNameInput, 'Misafir');

    await user.click(screen.getByRole('button', { name: /^Kaydet$/i }));

    await waitFor(() => {
      expect(mockCreateGuest).toHaveBeenCalled();
    });

    const [payload, apiKey] = mockCreateGuest.mock.calls[0];
    expect(payload.first_name).toBe('Yeni');
    expect(payload.last_name).toBe('Misafir');
    expect(apiKey).toBe('test-key');
  });
});

